#include "model.hpp"
#include "card_check.hpp"
#include "env.hpp"
#include "kernels.hpp"
#include "rope_gguf.hpp"

#include "store.hpp"

#include <cuda_runtime.h>

#include <sys/mman.h>
#include <sys/stat.h>

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <type_traits>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <thread>

namespace fnx {

namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

// a card's PCI address in sysfs's spelling
std::string card_bdf(int dev) {
    char bus[32];
    ck(cudaDeviceGetPCIBusId(bus, sizeof bus, dev), "cudaDeviceGetPCIBusId");
    try {
        return sysfs_bdf(bus);
    } catch (const std::exception& e) {
        throw std::runtime_error("card " + std::to_string(dev) + ": " + e.what());
    }
}
// the ring stores straight into the next card's memory: with translated IOMMU domains every such write faults
// (dmesg: "DMAR: ... PTE Write access is not set") and the first window ends in a ring timeout that names nothing.
// Each card's IOMMU group must be identity (boot with iommu=pt); a card whose iommu_group link is absent has no IOMMU
// in the way. Only absent: a link that cannot be read or points nowhere (iommu_groups hidden from this process) is
// refused, as a translated domain behind it would look the same
void require_identity_iommu(int dev, const std::string& dir) {
    if (access(dir.c_str(), F_OK) != 0)
        throw std::runtime_error("card " + std::to_string(dev) + ": no " + dir + " to read its IOMMU group from");
    const std::string link = dir + "/iommu_group", path = link + "/type";
    struct stat st;
    if (lstat(link.c_str(), &st) != 0) {
        if (errno == ENOENT) return;
        throw std::runtime_error("card " + std::to_string(dev) + ": cannot read " + link + ": " + std::strerror(errno));
    }
    std::FILE* f = std::fopen(path.c_str(), "r");
    char type[64] = {0};
    if (!f || !std::fgets(type, sizeof type, f)) {
        if (f) std::fclose(f);
        throw std::runtime_error("card " + std::to_string(dev) + ": cannot read " + path);
    }
    std::fclose(f);
    std::string t(type);
    while (!t.empty() && (t.back() == '\n' || t.back() == ' ')) t.pop_back();
    if (t != "identity")
        throw std::runtime_error("card " + std::to_string(dev) + " is in an IOMMU group of type '" + t + "' (" + path +
                                 "): the ring's writes between cards need identity domains; boot with iommu=pt");
}
// the card's BAR1 from sysfs, held to its memory (require_bar1_covers)
void check_bar1(int dev, const std::string& bdf, const std::string& dir, const cudaDeviceProp& prop) {
    const std::string path = dir + "/resource";
    std::FILE* f = std::fopen(path.c_str(), "r");
    if (!f) throw std::runtime_error("card " + std::to_string(dev) + ": cannot open " + path + " to read its BAR1 from");
    std::string text;
    char buf[4096];
    for (size_t n; (n = std::fread(buf, 1, sizeof buf, f)) > 0;) text.append(buf, n);
    const bool bad = std::ferror(f);
    std::fclose(f);
    if (bad) throw std::runtime_error("card " + std::to_string(dev) + ": cannot read " + path);
    require_bar1_covers(dev, bdf, prop.name, bar1_bytes(text, path), prop.totalGlobalMem, NCARD);
}
// the Makefile passes the arch the kernels were compiled for (ARCH): a default here would let a card of another one
// run them through the driver's PTX compiler unchecked
#ifndef FNX_ARCH
#error "FNX_ARCH is unset: build through the Makefile (make ARCH=70 or 86)"
#endif
// The cards are checked before anything is placed on them (src/card_check.hpp has the rules): a card that fails one
// would otherwise take the whole load first and then fail far from the cause, an sm_70 build on 24 GB cards out of
// memory, a short BAR1 as a ring that never completes. The IOMMU check comes before BAR1's, as it is the one that
// refuses a card with no sysfs directory: BAR1's would only say it cannot open the file
void check_cards() {
    const bool skip_bar1 = env_flag("FNX_SKIP_BAR1_CHECK");
    if (skip_bar1) std::fprintf(stderr, "fnx: FNX_SKIP_BAR1_CHECK=1: the cards' BAR1 is not checked against their memory\n");
    for (int dev = 0; dev < NCARD; ++dev) {
        cudaDeviceProp prop;
        ck(cudaGetDeviceProperties(&prop, dev), "cudaGetDeviceProperties");
        require_card_arch(dev, prop.name, prop.major, prop.minor, FNX_ARCH);
        const std::string bdf = card_bdf(dev), dir = "/sys/bus/pci/devices/" + bdf;
        require_identity_iommu(dev, dir);
        if (!skip_bar1) check_bar1(dev, bdf, dir, prop);
    }
}

bool lmhead_6p2_enabled() { return env_int("FNX_LMHEAD_6P2", 1) != 0; }

#if FNX_27B
int mtp_fasthead_clusters() {
    const long long v = env_int("FNX_MTP_FASTHEAD", 64);
    if (v < 0) throw std::runtime_error("FNX_MTP_FASTHEAD: want >= 0, got " + std::to_string(v));
    if (v == 1) return 64;
    return (int) v;
}

std::string resolve_vhead_clusters_path(const std::string& gguf_path) {
    if (const char* ep = std::getenv("FNX_VHEAD_CLUSTERS")) {
        if (!*ep || !std::strcmp(ep, "none")) return "";
        if (access(ep, R_OK) != 0) throw std::runtime_error(std::string("FNX_VHEAD_CLUSTERS: cannot read ") + ep);
        return ep;
    }
    const size_t slash = gguf_path.find_last_of('/');
    const std::string dir_cand = (slash != std::string::npos ? gguf_path.substr(0, slash + 1) : "") + "qwen38_vhead_clusters_1024.bin";
    if (access(dir_cand.c_str(), R_OK) == 0) return dir_cand;
    if (access("serve/qwen38_vhead_clusters_1024.bin", R_OK) == 0) return "serve/qwen38_vhead_clusters_1024.bin";
    return "";
}

struct CardClusterSlice {
    int n_clusters = 0;
    std::vector<int32_t> perm_to_vocab, vocab_to_local, cluster_off;
    std::vector<uint16_t> cent_f16;
};

void load_vhead_clusters_for_card(const std::string& path, int expected_n_in, int v0, int v1, CardClusterSlice& dst) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("vhead clusters: cannot open " + path);
    auto rd = [&](void* p, size_t b) {
        if (b && std::fread(p, 1, b, f) != b) {
            std::fclose(f);
            throw std::runtime_error("vhead clusters: short read in " + path);
        }
    };
    char magic[4] = {};
    int32_t hdr[4] = {};
    rd(magic, 4);
    rd(hdr, sizeof(hdr));
    if (std::memcmp(magic, "VHC1", 4) != 0) {
        std::fclose(f);
        throw std::runtime_error("vhead clusters: bad magic in " + path);
    }
    if (hdr[0] != NCARD || hdr[2] != expected_n_in) {
        std::fclose(f);
        throw std::runtime_error("vhead clusters: geometry mismatch in " + path + " (ncard=" + std::to_string(hdr[0]) +
                                 ", n_in=" + std::to_string(hdr[2]) + ")");
    }
    bool found = false;
    for (int c = 0; c < hdr[0]; ++c) {
        int32_t ch[3] = {};
        rd(ch, sizeof(ch));
        const int cv0 = ch[0], cv1 = ch[1], k = ch[2], n_out = cv1 - cv0;
        if (cv0 == v0 && cv1 == v1) {
            dst.n_clusters = k;
            dst.perm_to_vocab.resize(n_out);
            dst.vocab_to_local.resize(n_out);
            dst.cluster_off.resize(k + 1);
            dst.cent_f16.resize((size_t) k * hdr[2]);
            rd(dst.perm_to_vocab.data(), dst.perm_to_vocab.size() * 4);
            rd(dst.vocab_to_local.data(), dst.vocab_to_local.size() * 4);
            rd(dst.cluster_off.data(), dst.cluster_off.size() * 4);
            rd(dst.cent_f16.data(), dst.cent_f16.size() * 2);
            if (dst.cluster_off[0] != 0 || dst.cluster_off[k] != n_out) {
                std::fclose(f);
                throw std::runtime_error("vhead clusters: invalid cluster_off bounds in " + path);
            }
            for (int i = 0; i < n_out; ++i) {
                int p = dst.vocab_to_local[i];
                if (p < 0 || p >= n_out || dst.perm_to_vocab[p] != i) {
                    std::fclose(f);
                    throw std::runtime_error("vhead clusters: invalid permutation in " + path);
                }
            }
            found = true;
            break;
        }
        const long long skip = ((long long) n_out * 2 + (long long) (k + 1)) * 4LL + (long long) k * hdr[2] * 2LL;
        if (fseeko(f, skip, SEEK_CUR) != 0) {
            std::fclose(f);
            throw std::runtime_error("vhead clusters: seek failed in " + path);
        }
    }
    std::fclose(f);
    if (!found) throw std::runtime_error("vhead clusters: no slice [" + std::to_string(v0) + ", " + std::to_string(v1) + ") in " + path);
}
#endif

float f16_to_f32(uint16_t h) {
    const uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
    uint32_t u;
    if (e == 0) {
        if (m == 0) u = s << 31;
        else {
            int ee = 0;
            uint32_t mm = m;
            while (!(mm & 1024)) { mm <<= 1; --ee; }
            u = (s << 31) | ((uint32_t) (ee + 127 - 14) << 23) | ((mm & 1023) << 13);
        }
    } else if (e == 31) u = (s << 31) | 0x7f800000u | (m << 13);
    else u = (s << 31) | ((e + 112) << 23) | (m << 13);
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

uint16_t f32_to_f16(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    const uint32_t s = (u >> 16) & 0x8000u;
    const int e = (int) ((u >> 23) & 0xff) - 127 + 15;
    uint32_t m = u & 0x7fffffu;
    if (e <= 0) {   // subnormal or zero in f16
        if (e < -10) return (uint16_t) s;
        m |= 0x800000u;
        const int shift = 14 - e;
        uint32_t h = m >> shift;
        if ((m >> (shift - 1)) & 1u && ((m & ((1u << (shift - 1)) - 1)) || (h & 1u))) ++h;
        return (uint16_t) (s | h);
    }
    if (e >= 31) return (uint16_t) (s | 0x7c00u);
    uint32_t h = ((uint32_t) e << 10) | (m >> 13);
    if ((m >> 12) & 1u && ((m & 0xfffu) || (h & 1u))) ++h;   // nearest, ties to even
    return (uint16_t) (s | h);
}

uint16_t f32_to_bf16(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    return (uint16_t) ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

// an F32 tensor's first n values, read on the host
const float* f32_data(const Tensor& t, size_t n) {
    if (t.type != T_F32) throw std::runtime_error(t.name + " is not F32");
    if (t.nbytes < n * sizeof(float)) throw std::runtime_error(t.name + " holds fewer than " + std::to_string(n) + " values");
    return (const float*) t.data;
}

// a tensor's range of the mapping locked, LOCKERS threads each faulting in a part: one thread faults a cold file in
// readahead-sized pieces (1.1 GB/s from the NVMe), 16 side by side reached 2.3 (tools/stage_test.cpp)
void lock_in_ram(const Tensor& t) {
    constexpr int LOCKERS = 16;
    constexpr uintptr_t PG = 4096;
    const uintptr_t a = (uintptr_t) t.data / PG * PG, e = (uintptr_t) t.data + t.nbytes;
    const uintptr_t part = ((e - a) / LOCKERS + PG - 1) / PG * PG;
    std::vector<int> err(LOCKERS, 0);
    std::vector<std::thread> th;
    for (int i = 0; i < LOCKERS; ++i)
        th.emplace_back([&, i] {
            const uintptr_t lo = std::min(e, a + (uintptr_t) i * part), hi = std::min(e, lo + part);
            if (hi > lo && mlock((const void*) lo, hi - lo) != 0) {
                err[(size_t) i] = errno;
                // Z420 sm61-port, FNX_MLOCK_SOFT=1: fault the range in instead (the page cache keeps it on a box with
                // RAM to spare); the warning below says so
                if (env_flag("FNX_MLOCK_SOFT")) {
                    unsigned sink = 0;
                    for (uintptr_t p = lo; p < hi; p += PG) sink += *(const volatile unsigned char*) p;
                    (void) sink;
                }
            }
        });
    for (auto& x : th) x.join();
    for (int en : err)
        if (en && env_flag("FNX_MLOCK_SOFT")) {
            std::fprintf(stderr, "fnx: mlock of %s (%zu MiB) failed: %s; FNX_MLOCK_SOFT=1: faulted in, not locked\n", t.name.c_str(),
                         (size_t) (t.nbytes >> 20), std::strerror(en));
            return;
        } else if (en)
            throw std::runtime_error("fnx: mlock of " + t.name + " (" + std::to_string(t.nbytes >> 20) + " MiB) failed: " + std::strerror(en) +
                                     "; run with RLIMIT_MEMLOCK raised (systemd-run -p LimitMEMLOCK=infinity)");
}

// the stager's groups for load_card: group l every tensor of layer l, group NL the output head's. The head is read
// through the page cache: the draft layer (Mtp) takes its rows of output.weight from the mapping after the load
std::vector<Stager::Group> stage_groups(const Gguf& g) {
#if FNX_27B
    // and group NL + 1 the draft layer, blk.NL (Flash-Next's comes from separate files, Mtp)
    std::vector<Stager::Group> gs(NL + 2);
    for (const auto& kv : g.tensors()) {
        int l = -1, n = 0;
        if (std::sscanf(kv.first.c_str(), "blk.%d.%n", &l, &n) == 1 && n > 0 && l >= 0 && l <= NL)
            gs[(size_t) (l < NL ? l : NL + 1)].names.push_back(kv.first);
    }
    gs[NL].names = {"output_norm.weight", "output.weight"};
#else
    std::vector<Stager::Group> gs(NL + 1);
    for (const auto& kv : g.tensors()) {
        int l = -1, n = 0;
        if (std::sscanf(kv.first.c_str(), "blk.%d.%n", &l, &n) == 1 && n > 0 && l >= 0 && l < NL) gs[(size_t) l].names.push_back(kv.first);
    }
    gs[NL].names = {"output_hc_norm.weight", "output_hc_down.weight", "output_hc_up.weight", "output.weight"};
#endif
    gs[NL].cached = true;
    return gs;
}

// the column slicers take block indices that the callers count in 32-value blocks (ff0 / 32, HS / 32, HD / 32)
TypeInfo block32(const Tensor& t) {
    const TypeInfo ti = type_info(t.type);
    if (ti.block != 32) throw std::runtime_error(t.name + ": sliced by columns, which needs 32-value blocks, not " + std::to_string(ti.block));
    return ti;
}

struct Up {
    CardW& c;
    Store* store = nullptr;               // the lab's holder: the shared memory instead of cudaMalloc / cudaHostAlloc
    std::vector<void*>* track = nullptr;  // a worker placing its experts: what it allocates (Model::free_experts)
    template <typename T> T* put(const void* host, size_t bytes) {
        void* d = nullptr;
        if (store) d = store->dev(c.dev, bytes);
        else ck(cudaMalloc(&d, bytes), "cudaMalloc");
        if (track) track->push_back(d);
        ck(cudaMemcpy(d, host, bytes, cudaMemcpyDefault), "upload");
        c.bytes += bytes;
        return (T*) d;
    }
    // a norm as the window reads it: exactly n floats. A tensor of another width would feed garbage rows silently;
    // the whole tensor must match, so extra rows fail too (the window never reads them)
    float* f32(const Tensor& t, size_t n) {
        if (t.type != T_F32) throw std::runtime_error(t.name + " is not F32");
        if (t.nbytes != n * 4)
            throw std::runtime_error(t.name + " holds " + std::to_string(t.nbytes / 4) + " floats, not the " + std::to_string(n) +
                                     " the window reads it at");
        return put<float>(t.data, t.nbytes);
    }
    // where a tensor's routed experts are assembled: with an ExpertCache (host_of) their mapped page-locked host copy,
    // else tmp
    uint8_t* expert_buf(size_t bytes, const char** host_of, std::vector<uint8_t>& tmp) {
        if (!host_of) {
            tmp.resize(bytes);
            return tmp.data();
        }
        void* hp = nullptr;
        const cudaError_t err = store ? (hp = store->host(c.dev, bytes), cudaSuccess) : cudaHostAlloc(&hp, bytes, cudaHostAllocMapped);
        if (err != cudaSuccess)
            throw std::runtime_error("place: " + std::to_string(bytes >> 20) + " MiB of page-locked host memory for experts: " +
                                     cudaGetErrorString(err));
        c.host_bytes += bytes;
        return (uint8_t*) hp;
    }
    // routed experts (h from expert_buf: every expert's matrix, stride bytes apart): to VRAM at p, or with an
    // ExpertCache each one's address in host_of (the layer's row of the card's host addresses)
    void place(DW& w, const uint8_t* h, size_t bytes, size_t stride, const char** host_of) {
        if (!host_of) {
            w.p = put<uint8_t>(h, bytes);
            return;
        }
        void* dp = nullptr;
        if (store) dp = (void*) h;   // the shared host memory has one address for the CPU and the cards
        else ck(cudaHostGetDevicePointer(&dp, (void*) h, 0), "place: mapped pointer");
        for (size_t e = 0; e < bytes / stride; ++e) host_of[e] = (const char*) dp + e * stride;
    }
    // the card's ExpertCache: a slot for each resident expert (res) in its size's pool, filled from its host copy; the
    // layers' tables. With arena_score (FNX_ARENA) the largest pool's slots go coldest first and are neither allocated
    // nor filled here: the engine puts them in its arena (Engine::arena_fill); until then its experts point at RAM
    void expert_cache(const std::vector<std::vector<char>>& res, const std::vector<const char*>& hgu, const std::vector<const char*>& hdn,
                      const std::vector<double>* arena_score) {
        ExpertCache& ec = c.ec;
        std::vector<std::pair<size_t, size_t>> sizes;   // each pool's (gate|up, down) bytes
        for (int l = 0; l < NL; ++l) {
            const LayerW& L = c.L[(size_t) l];
            if (L.gu_e_stride % 16 || L.down_e_stride % 16) throw std::runtime_error("expert cache: expert strides not in 16-byte units");
            const std::pair<size_t, size_t> s{L.gu_e_stride, L.down_e_stride};
            auto it = std::find(sizes.begin(), sizes.end(), s);
            if (it == sizes.end()) {
                if (sizes.size() == EC_POOLS) throw std::runtime_error("expert cache: more than " + std::to_string(EC_POOLS) + " expert sizes");
                sizes.push_back(s);
                it = sizes.end() - 1;
            }
            ec.pool_of[l] = (int) (it - sizes.begin());
        }
        ec.n_pools = (int) sizes.size();
        if (arena_score) {   // the largest pool by bytes
            size_t most = 0;
            for (int p = 0; p < ec.n_pools; ++p) {
                size_t b = 0;
                for (int l = 0; l < NL; ++l) b += ec.pool_of[l] == p ? (size_t) NEXP * (sizes[(size_t) p].first + sizes[(size_t) p].second) : 0;
                if (b > most) most = b, ec.arena_pool = p;
            }
        }
        std::vector<int> owner, slot_of((size_t) NL * NEXP, -1);
        for (int p = 0; p < ec.n_pools; ++p) {
            ec.s0[p] = (int) owner.size();
            for (int l = 0; l < NL; ++l)
                if (ec.pool_of[l] == p)
                    for (int e = 0; e < NEXP; ++e)
                        if (res[(size_t) l][(size_t) e]) owner.push_back(l * NEXP + e);
            if (p == ec.arena_pool)
                std::stable_sort(owner.begin() + ec.s0[p], owner.end(),
                                 [&](int a, int b) { return (*arena_score)[(size_t) a] < (*arena_score)[(size_t) b]; });
            for (int s = ec.s0[p]; s < (int) owner.size(); ++s) slot_of[(size_t) owner[(size_t) s]] = s;
            ec.n[p] = (int) owner.size() - ec.s0[p];
            if (ec.n[p] < 2 * MAXT * TOPK)
                throw std::runtime_error("expert cache: pool " + std::to_string(p) + " has " + std::to_string(ec.n[p]) + " slots, a window needs " +
                                         std::to_string(2 * MAXT * TOPK) + " (fewer experts in RAM)");
            ec.dn_off[p] = sizes[(size_t) p].first;
            ec.slot_bytes[p] = sizes[(size_t) p].first + sizes[(size_t) p].second;
            if (p == ec.arena_pool) continue;
            ck(cudaMalloc(&ec.pool[p], (size_t) ec.n[p] * ec.slot_bytes[p]), "expert cache: slots");
            if (track) track->push_back(ec.pool[p]);
            c.bytes += (size_t) ec.n[p] * ec.slot_bytes[p];
        }
        ec.n_slots = (int) owner.size();
        ec.arena_full = arena_score && ec.n_slots == NL * NEXP;
        std::vector<const char*> epg(hgu), epd(hdn);
        for (int s = 0; s < ec.n_slots; ++s) {
            const int i = owner[(size_t) s], p = ec.pool_of[i / NEXP];
            if (p == ec.arena_pool) continue;
            const LayerW& L = c.L[(size_t) (i / NEXP)];
            char* at = ec.pool[p] + (size_t) (s - ec.s0[p]) * ec.slot_bytes[p];
            ck(cudaMemcpy(at, hgu[(size_t) i], L.gu_e_stride, cudaMemcpyDefault), "expert cache: fill");
            ck(cudaMemcpy(at + ec.dn_off[p], hdn[(size_t) i], L.down_e_stride, cudaMemcpyDefault), "expert cache: fill");
            epg[(size_t) i] = at;
            epd[(size_t) i] = at + ec.dn_off[p];
        }
        ec.ep_gu = put<const char*>(epg.data(), epg.size() * sizeof(char*));
        ec.ep_dn = put<const char*>(epd.data(), epd.size() * sizeof(char*));
        ec.host_gu = put<const char*>(hgu.data(), hgu.size() * sizeof(char*));
        ec.host_dn = put<const char*>(hdn.data(), hdn.size() * sizeof(char*));
        c.ec_host_gu = hgu;
        c.ec_host_dn = hdn;
        ec.slot_of = put<int>(slot_of.data(), slot_of.size() * sizeof(int));
        c.ec_owner0 = owner;
        c.ec_slot0 = slot_of;
        ec.owner = put<int>(owner.data(), owner.size() * sizeof(int));
        const std::vector<unsigned char> ref((size_t) ec.n_slots, 0);
        ec.ref = put<unsigned char>(ref.data(), ref.size());
        ec.hand = put<int>(ec.s0, sizeof ec.s0);   // each pool's hand at its first slot
        const int none[EC_POOLS] = {};
        ec.lent = put<int>(none, sizeof none);
        for (int l = 0; l < NL; ++l) {
            c.L[(size_t) l].gu_e.ep = ec.ep_gu + (size_t) l * NEXP;
            c.L[(size_t) l].down_e.ep = ec.ep_dn + (size_t) l * NEXP;
        }
    }
    // row r of t (counting every matrix's rows, in order), which must be one of its rows: a row list made from the model's
    // geometry would otherwise read the next tensor's bytes from the mapping when a GGUF's tensor is shorter than it
    static const uint8_t* row_of(const Tensor& t, int r, size_t row_bytes) {
        if (r < 0 || r >= t.nrows())
            throw std::runtime_error(t.name + ": row " + std::to_string(r) + " asked for, it has " + std::to_string(t.nrows()));
        return t.data + (size_t) r * row_bytes;
    }
    // rows `rows` of a 2-D F32 tensor (ne1 indexes rows)
    float* f32_rows(const Tensor& t, const std::vector<int>& rows, size_t row_floats) {
        if (t.type != T_F32) throw std::runtime_error(t.name + " is not F32");
        if ((size_t) t.ne[0] != row_floats)
            throw std::runtime_error(t.name + " is " + std::to_string(t.ne[0]) + " floats a row, not the " + std::to_string(row_floats) +
                                     " its reader takes");
        std::vector<float> h(rows.size() * (size_t) t.ne[0]);
        for (size_t i = 0; i < rows.size(); ++i)
            std::memcpy(&h[i * t.ne[0]], row_of(t, rows[i], (size_t) t.ne[0] * 4), (size_t) t.ne[0] * 4);
        return put<float>(h.data(), h.size() * 4);
    }
    // Q8_0 -> bf16, the whole tensor; F32 -> bf16 likewise
    uint16_t* bf16(const Tensor& t) {
        const size_t n = (size_t) (t.ne[0] * t.ne[1] * t.ne[2] * t.ne[3]);
        std::vector<uint16_t> h(n);
        if (t.type == T_F32) {
            const float* f = (const float*) t.data;
            for (size_t i = 0; i < n; ++i) h[i] = f32_to_bf16(f[i]);
        } else if (t.type == T_Q8_0) {
            for (size_t b = 0; b < n / 32; ++b) {
                const uint8_t* blk = t.data + b * 34;
                uint16_t dh;
                std::memcpy(&dh, blk, 2);
                const float d = f16_to_f32(dh);
                for (int j = 0; j < 32; ++j) h[b * 32 + j] = f32_to_bf16(d * (float) (int8_t) blk[2 + j]);
            }
        } else if (t.type == T_BF16) {
            std::memcpy(h.data(), t.data, n * 2);
        } else {
            throw std::runtime_error(t.name + ": bf16 conversion supports F32, Q8_0 and BF16");
        }
        return put<uint16_t>(h.data(), h.size() * 2);
    }
    // Z420 sm61-port: split Q8_0 slices directly on the host before uploading, avoiding 58 GB of H2D->D2H->cudaFree round-trips
    DW finish_slice(DW w, const std::vector<uint8_t>& h) {
        if (w.type == T_Q8_0) {
            const size_t nblk = h.size() / 34;
            std::vector<int8_t> q(nblk * 32);
            std::vector<uint16_t> d(nblk);
            for (size_t b = 0; b < nblk; ++b) {
                std::memcpy(&d[b], &h[b * 34], 2);
                std::memcpy(&q[b * 32], &h[b * 34 + 2], 32);
            }
            w.p = nullptr;
            w.s.q = put<int8_t>(q.data(), q.size());
            w.s.d = put<uint16_t>(d.data(), d.size() * 2);
            return w;
        }
        w.p = put<uint8_t>(h.data(), h.size());
        return w;
    }
    // a dense matrix for q8s_gemv: Q8_0 slices move to the split layout (other types stay as they are)
    DW dense(DW w) {
        if (w.type != T_Q8_0 || w.s.q) return w;
        const size_t bytes = (size_t) w.n_out * w.row_bytes;
        std::vector<uint8_t> h(bytes);
        ck(cudaMemcpy(h.data(), w.p, bytes, cudaMemcpyDeviceToHost), "dense: download");
        if (store) store->free_dev(c.dev, w.p, bytes);   // the upload just before: its room is used again
        else ck(cudaFree(w.p), "dense: free");
        if (track) track->erase(std::remove(track->begin(), track->end(), (void*) w.p), track->end());   // not freed twice
        c.bytes -= bytes;
        return finish_slice(w, h);
    }
    VHead6p2 vhead_6p2(const Tensor& t, int r0, int r1, const int32_t* perm = nullptr, const int32_t* inv = nullptr) {
        if (t.type != T_Q8_0) throw std::runtime_error(t.name + ": vhead_6p2 needs Q8_0");
        if (r0 < 0 || r1 <= r0 || (int64_t) r1 > t.ne[1])
            throw std::runtime_error(t.name + ": vhead_6p2 row range out of bounds");
        const int rows = r1 - r0, cols = (int) t.ne[0];
        if (cols <= 0 || (cols & 511) != 0)
            throw std::runtime_error(t.name + ": vhead_6p2 needs cols a positive multiple of 512");
        const size_t rb = t.row_bytes();
        std::vector<uint8_t> raw((size_t) rows * rb);
        for (int i = 0; i < rows; ++i) std::memcpy(&raw[(size_t) i * rb], row_of(t, r0 + i, rb), rb);
        const int S = cols / 512, nb = cols / 32;
        std::vector<uint32_t> hi((size_t) rows * S * 96);
        std::vector<uint32_t> lo((size_t) rows * S * 32);
        std::vector<uint16_t> d((size_t) rows * nb);
        std::vector<float2> row_meta((size_t) rows);
        pack_q80_to_6p2_host(raw.data(), rb, rows, cols, perm, hi.data(), lo.data(), d.data(), row_meta.data());
        VHead6p2 vh;
        vh.n_in = cols;
        vh.n_out = rows;
        vh.hi = put<uint32_t>(hi.data(), hi.size() * sizeof(uint32_t));
        vh.lo = put<uint32_t>(lo.data(), lo.size() * sizeof(uint32_t));
        vh.d = put<uint16_t>(d.data(), d.size() * sizeof(uint16_t));
        vh.row_meta = put<float2>(row_meta.data(), row_meta.size() * sizeof(float2));
        if (perm) vh.perm_to_vocab = put<int32_t>(perm, (size_t) rows * sizeof(int32_t));
        if (inv) vh.vocab_to_local = put<int32_t>(inv, (size_t) rows * sizeof(int32_t));
        return vh;
    }
    // floating values [n] as the split int8 layout (per 32 values: d = amax / 127, ggml's rounding)
    template <class F>
    Q8S q8split_vals(size_t n, F get, const char* tag) {
        if (n % 32 != 0) throw std::runtime_error(std::string(tag) + ": q8split needs length multiple of 32");
        std::vector<int8_t> q(n);
        std::vector<uint16_t> d(n / 32);
        for (size_t b = 0; b < n / 32; ++b) {
            float x[32], amax = 0.f;
            for (int j = 0; j < 32; ++j) {
                x[j] = get(b * 32 + j);
                amax = std::max(amax, std::fabs(x[j]));
            }
            const float dd = amax / 127.0f, id = dd > 0.f ? 1.0f / dd : 0.f;
            for (int j = 0; j < 32; ++j) q[b * 32 + j] = (int8_t) std::lround(x[j] * id);
            d[b] = f32_to_f16(dd);
        }
        Q8S s;
        s.q = put<int8_t>(q.data(), q.size());
        s.d = put<uint16_t>(d.data(), d.size() * 2);
        return s;
    }
    // bf16 values [n] as the split int8 layout (per 32 values: d = amax / 127, ggml's rounding)
    Q8S q8split_bf16(const uint16_t* v, size_t n) {
        return q8split_vals(
            n,
            [&](size_t idx) {
                const uint32_t u = (uint32_t) v[idx] << 16;
                float f;
                std::memcpy(&f, &u, 4);
                return f;
            },
            "q8split_bf16");
    }
    // a whole Q8_0 tensor as int8 values and f16 scales
    Q8S q8split(const Tensor& t) {
        if (t.type != T_Q8_0) throw std::runtime_error(t.name + ": q8split needs Q8_0");
        const size_t n = (size_t) (t.ne[0] * t.ne[1] * t.ne[2] * t.ne[3]);
        std::vector<int8_t> q(n);
        std::vector<uint16_t> d(n / 32);
        for (size_t b = 0; b < n / 32; ++b) {
            const uint8_t* blk = t.data + b * 34;
            std::memcpy(&d[b], blk, 2);
            std::memcpy(&q[b * 32], blk + 2, 32);
        }
        Q8S s;
        s.q = put<int8_t>(q.data(), q.size());
        s.d = put<uint16_t>(d.data(), d.size() * 2);
        return s;
    }
    // Q8_0, BF16, or F32 tensor as the split int8+f16 layout (hc8)
    Q8S q8split_any(const Tensor& t) {
        const size_t n = (size_t) t.ne[0] * t.ne[1] * t.ne[2] * t.ne[3];
        if (t.type == T_Q8_0) {
            if (n % 32 != 0) throw std::runtime_error(t.name + ": q8split needs length multiple of 32");
            return q8split(t);
        }
        if (t.type == T_BF16) return q8split_bf16(reinterpret_cast<const uint16_t*>(t.data), n);
        if (t.type == T_F32) {
            return q8split_vals(
                n,
                [&](size_t idx) {
                    float f;
                    std::memcpy(&f, t.data + idx * 4, 4);
                    return f;
                },
                t.name.c_str());
        }
        throw std::runtime_error(t.name + ": q8split_any needs Q8_0, BF16, or F32");
    }
    // whole rows of a quantized 2-D tensor (or of matrix `mat` of a 3-D one)
    DW rows(const Tensor& t, const std::vector<int>& rows) {
        DW w;
        w.type = t.type;
        w.n_in = (int) t.ne[0];
        w.n_out = (int) rows.size();
        w.row_bytes = t.row_bytes();
        std::vector<uint8_t> h(rows.size() * w.row_bytes);
        for (size_t i = 0; i < rows.size(); ++i) std::memcpy(&h[i * w.row_bytes], row_of(t, rows[i], w.row_bytes), w.row_bytes);
        return finish_slice(w, h);
    }
    // rows r1 of t1 then rows r2 of t2 (same type and row width), one matrix
    DW rows2(const Tensor& t1, const std::vector<int>& r1, const Tensor& t2, const std::vector<int>& r2) {
        if (t1.type != t2.type || t1.ne[0] != t2.ne[0]) throw std::runtime_error(t1.name + " and " + t2.name + " do not stack");
        DW w;
        w.type = t1.type;
        w.n_in = (int) t1.ne[0];
        w.n_out = (int) (r1.size() + r2.size());
        w.row_bytes = t1.row_bytes();
        std::vector<uint8_t> h((size_t) w.n_out * w.row_bytes);
        size_t o = 0;
        for (int r : r1) { std::memcpy(&h[o], row_of(t1, r, w.row_bytes), w.row_bytes); o += w.row_bytes; }
        for (int r : r2) { std::memcpy(&h[o], row_of(t2, r, w.row_bytes), w.row_bytes); o += w.row_bytes; }
        return finish_slice(w, h);
    }
#if FNX_27B
    // rows r1 of t1, r2 of t2, then r3 of t3, one matrix
    DW rows3(const Tensor& t1, const std::vector<int>& r1, const Tensor& t2, const std::vector<int>& r2, const Tensor& t3,
             const std::vector<int>& r3) {
        for (const Tensor* t : {&t2, &t3})
            if (t1.type != t->type || t1.ne[0] != t->ne[0]) throw std::runtime_error(t1.name + " and " + t->name + " do not stack");
        DW w;
        w.type = t1.type;
        w.n_in = (int) t1.ne[0];
        w.n_out = (int) (r1.size() + r2.size() + r3.size());
        w.row_bytes = t1.row_bytes();
        std::vector<uint8_t> h((size_t) w.n_out * w.row_bytes);
        size_t o = 0;
        for (const auto& [t, rs] : {std::pair{&t1, &r1}, std::pair{&t2, &r2}, std::pair{&t3, &r3}})
            for (int r : *rs) { std::memcpy(&h[o], row_of(*t, r, w.row_bytes), w.row_bytes); o += w.row_bytes; }
        return finish_slice(w, h);
    }
#endif
    // rows `rows` of t1 then of t2 as f32 (F32 rows as they are, Q8_0 rows dequantized)
    float* f32_rows2(const Tensor& t1, const Tensor& t2, const std::vector<int>& rows, size_t row_floats) {
        std::vector<float> h;
        for (const Tensor* t : {&t1, &t2}) {
            const size_t n = (size_t) t->ne[0];
            if (n != row_floats)
                throw std::runtime_error(t->name + " is " + std::to_string(n) + " floats a row, not the " + std::to_string(row_floats) +
                                         " its reader takes");
            for (int r : rows) {
                if (t->type == T_F32) {
                    const float* row = (const float*) row_of(*t, r, n * 4);
                    h.insert(h.end(), row, row + n);
                } else if (t->type == T_Q8_0 && n % 32 == 0) {
                    const uint8_t* row = row_of(*t, r, t->row_bytes());
                    for (size_t b = 0; b < n / 32; ++b) {
                        uint16_t dh;
                        std::memcpy(&dh, row + b * 34, 2);
                        const float d = f16_to_f32(dh);
                        for (int j = 0; j < 32; ++j) h.push_back(d * (float) (int8_t) row[b * 34 + 2 + j]);
                    }
                } else {
                    throw std::runtime_error(t->name + " is neither F32 nor Q8_0");
                }
            }
        }
        return put<float>(h.data(), h.size() * 4);
    }
    // every row, keeping input blocks `blocks` (block indices along ne0) of each: a column slice
    DW cols(const Tensor& t, const std::vector<int>& blocks) {
        const TypeInfo ti = block32(t);
        DW w;
        w.type = t.type;
        w.n_in = (int) blocks.size() * ti.block;
        w.n_out = (int) t.ne[1];
        w.row_bytes = blocks.size() * ti.bytes;
        const size_t src_row = t.row_bytes();
        for (int b : blocks)
            if (b < 0 || (size_t) (b + 1) * ti.bytes > src_row)
                throw std::runtime_error(t.name + ": block " + std::to_string(b) + " asked for, a row has " + std::to_string(src_row / ti.bytes));
        std::vector<uint8_t> h((size_t) w.n_out * w.row_bytes);
        for (int r = 0; r < w.n_out; ++r)
            for (size_t b = 0; b < blocks.size(); ++b)
                std::memcpy(&h[(size_t) r * w.row_bytes + b * ti.bytes], t.data + (size_t) r * src_row + (size_t) blocks[b] * ti.bytes, ti.bytes);
        return finish_slice(w, h);
    }
    // per expert: rows [r0, r1) (gate / up), expert-major
    static void row_range(const Tensor& t, int r0, int r1) {
        if (r0 < 0 || r1 <= r0 || r1 > t.ne[1])
            throw std::runtime_error(t.name + ": rows [" + std::to_string(r0) + ", " + std::to_string(r1) + ") asked for, it has " +
                                     std::to_string(t.ne[1]) + " a matrix");
    }
    DW expert_rows(const Tensor& t, int r0, int r1, size_t& stride) {
        row_range(t, r0, r1);
        DW w;
        w.type = t.type;
        w.n_in = (int) t.ne[0];
        w.n_out = r1 - r0;
        w.row_bytes = t.row_bytes();
        stride = (size_t) w.n_out * w.row_bytes;
        const size_t src_mat = (size_t) t.ne[1] * w.row_bytes;
        std::vector<uint8_t> h((size_t) t.ne[2] * stride);
        for (int64_t e = 0; e < t.ne[2]; ++e) std::memcpy(&h[e * stride], t.data + e * src_mat + (size_t) r0 * w.row_bytes, stride);
        w.p = put<uint8_t>(h.data(), h.size());
        return w;
    }
    // per expert: rows [r0, r1) of t1 then of t2 (gate; up), expert-major
    DW expert_rows2(const Tensor& t1, const Tensor& t2, int r0, int r1, size_t& stride, const char** host_of = nullptr) {
        if (t1.type != t2.type || t1.ne[0] != t2.ne[0] || t1.ne[2] != t2.ne[2]) throw std::runtime_error(t1.name + " and " + t2.name + " do not stack");
        row_range(t1, r0, r1);
        row_range(t2, r0, r1);
        DW w;
        w.type = t1.type;
        w.n_in = (int) t1.ne[0];
        w.n_out = 2 * (r1 - r0);
        w.row_bytes = t1.row_bytes();
        stride = (size_t) w.n_out * w.row_bytes;
        const size_t half = (size_t) (r1 - r0) * w.row_bytes, src_mat = (size_t) t1.ne[1] * w.row_bytes;
        std::vector<uint8_t> tmp;
        const size_t bytes = (size_t) t1.ne[2] * stride;
        uint8_t* h = expert_buf(bytes, host_of, tmp);
        for (int64_t e = 0; e < t1.ne[2]; ++e) {
            std::memcpy(&h[e * stride], t1.data + e * src_mat + (size_t) r0 * w.row_bytes, half);
            std::memcpy(&h[e * stride + half], t2.data + e * src_mat + (size_t) r0 * w.row_bytes, half);
        }
        place(w, h, bytes, stride, host_of);
        return w;
    }
    // f32 tensors' rows one after another
    float* f32_cat(const Tensor& t1, const Tensor& t2, size_t row_floats) {
        if (t1.type != T_F32 || t2.type != T_F32 || t1.ne[0] != t2.ne[0]) throw std::runtime_error(t1.name + " and " + t2.name + " do not stack");
        if ((size_t) t1.ne[0] != row_floats)
            throw std::runtime_error(t1.name + " is " + std::to_string(t1.ne[0]) + " floats a row, not the " + std::to_string(row_floats) +
                                     " its reader takes");
        std::vector<float> h((const float*) t1.data, (const float*) t1.data + t1.ne[0] * t1.ne[1]);
        h.insert(h.end(), (const float*) t2.data, (const float*) t2.data + t2.ne[0] * t2.ne[1]);
        return put<float>(h.data(), h.size() * 4);
    }
    // per expert: every row, input blocks [b0, b1) (down), expert-major
    DW expert_cols(const Tensor& t, int b0, int b1, size_t& stride, const char** host_of = nullptr) {
        const TypeInfo ti = block32(t);
        DW w;
        w.type = t.type;
        w.n_in = (b1 - b0) * ti.block;
        w.n_out = (int) t.ne[1];
        w.row_bytes = (size_t) (b1 - b0) * ti.bytes;
        stride = (size_t) w.n_out * w.row_bytes;
        const size_t src_row = t.row_bytes(), src_mat = (size_t) t.ne[1] * src_row;
        if (b0 < 0 || b1 <= b0 || (size_t) b1 * ti.bytes > src_row)
            throw std::runtime_error(t.name + ": blocks [" + std::to_string(b0) + ", " + std::to_string(b1) + ") asked for, a row has " +
                                     std::to_string(src_row / ti.bytes));
        std::vector<uint8_t> tmp;
        const size_t bytes = (size_t) t.ne[2] * stride;
        uint8_t* h = expert_buf(bytes, host_of, tmp);
        for (int64_t e = 0; e < t.ne[2]; ++e)
            for (int r = 0; r < w.n_out; ++r)
                std::memcpy(&h[e * stride + (size_t) r * w.row_bytes], t.data + e * src_mat + (size_t) r * src_row + (size_t) b0 * ti.bytes, w.row_bytes);
        place(w, h, bytes, stride, host_of);
        return w;
    }
};

std::vector<int> range(int a, int b) {
    std::vector<int> v;
    for (int i = a; i < b; ++i) v.push_back(i);
    return v;
}

#if FNX_27B
// The 27B's layers on a card, from load_card's channel lists: the recurrent layers as Flash-Next's; attention split by
// heads (K/V too: each card keeps its own heads' cache, src/s27_attn.cuh), q | gate, K and V stacked in one matrix;
// the dense FFN by FF width; then the head, and the draft layer (blk.NL, stage group NL + 1). Every matrix q8s reads
// is Q8_0: the 27B's path has no other matvec
void load_layers_27b(Stager& st, CardW& c, Up& up, const std::vector<int>& conv_rows, const std::vector<int>& zrows,
                     const std::vector<int>& vblocks, const std::vector<int>& qrows, const std::vector<int>& kvrows,
                     const std::vector<int>& woblocks, const std::string& clusters_file) {
    auto L = [&](int l, const char* s) -> const Tensor& { return st.t(l, "blk." + std::to_string(l) + "." + s); };
    const std::vector<int>& vheads = c.vheads;
    auto q8 = [](const DW& d, const std::string& what) {
        if (!d.s.q) throw std::runtime_error("27b: " + what + " is not Q8_0 (the 27B's matvecs read Q8_0 only)");
    };
    // the shape the window writes and reads it at (rows are the output stride; a wrong one would land at another)
    auto shape = [](const DW& d, int out, int in, const std::string& what) {
        if (d.n_out != out || d.n_in != in)
            throw std::runtime_error("27b: " + what + " is " + std::to_string(d.n_out) + " x " + std::to_string(d.n_in) + ", not " +
                                     std::to_string(out) + " x " + std::to_string(in));
    };
    const int ffs = c.fd1 - c.fd0, qkv = c.nq * 2 * HD + 2 * c.nkv * HD;
    auto attn = [&](LayerW& w, auto&& T) {
        w.wq = up.dense(up.rows3(T("attn_q.weight"), qrows, T("attn_k.weight"), kvrows, T("attn_v.weight"), kvrows));
        w.wo = up.dense(up.cols(T("attn_output.weight"), woblocks));
        w.q_norm = up.f32(T("attn_q_norm.weight"), HD);
        w.k_norm = up.f32(T("attn_k_norm.weight"), HD);
    };
    auto ffn = [&](LayerW& w, auto&& T) {
        w.attn_norm = up.f32(T("attn_norm.weight"), N);
        w.ffn_norm = up.f32(T("post_attention_norm.weight"), N);
        w.ffn_gu = up.dense(up.rows2(T("ffn_gate.weight"), range(c.fd0, c.fd1), T("ffn_up.weight"), range(c.fd0, c.fd1)));
        w.ffn_down = up.dense(up.cols(T("ffn_down.weight"), range(c.fd0 / 32, c.fd1 / 32)));
    };
    c.L.resize(NL);
    for (int l = 0; l < NL; ++l) {
        LayerW& w = c.L[l];
        auto T = [&](const char* s) -> const Tensor& { return L(l, s); };
        if (!is_attn(l)) {
            w.qkvz = up.dense(up.rows2(T("attn_qkv.weight"), conv_rows, T("attn_gate.weight"), zrows));
            w.ab = up.f32_rows2(T("ssm_alpha.weight"), T("ssm_beta.weight"), vheads, N);
            w.ssm_out = up.dense(up.cols(T("ssm_out.weight"), vblocks));
            w.conv = up.f32_rows(T("ssm_conv1d.weight"), conv_rows, 4);
            std::vector<float> a, dt;
            const size_t nh = (size_t) *std::max_element(vheads.begin(), vheads.end()) + 1;
            const float* af = f32_data(T("ssm_a"), nh);
            const float* df = f32_data(T("ssm_dt.bias"), nh);
            for (int h : vheads) { a.push_back(af[h]); dt.push_back(df[h]); }
            w.ssm_a = up.put<float>(a.data(), a.size() * 4);
            w.dt_bias = up.put<float>(dt.data(), dt.size() * 4);
            w.ssm_norm = up.f32(T("ssm_norm.weight"), HS);
            q8(w.qkvz, "layer " + std::to_string(l) + "'s qkv | gate"), q8(w.ssm_out, "layer " + std::to_string(l) + "'s ssm_out");
            shape(w.qkvz, c.conv_ch() + c.nv() * HS, N, "layer " + std::to_string(l) + "'s qkv | gate");
            shape(w.ssm_out, N, c.nv() * HS, "layer " + std::to_string(l) + "'s ssm_out");
        } else {
            attn(w, T);
            q8(w.wq, "layer " + std::to_string(l) + "'s q | k | v"), q8(w.wo, "layer " + std::to_string(l) + "'s attn_output");
            shape(w.wq, qkv, N, "layer " + std::to_string(l) + "'s q | k | v");
            shape(w.wo, N, c.nq * HD, "layer " + std::to_string(l) + "'s attn_output");
        }
        ffn(w, T);
        q8(w.ffn_gu, "layer " + std::to_string(l) + "'s gate | up"), q8(w.ffn_down, "layer " + std::to_string(l) + "'s down");
        shape(w.ffn_gu, 2 * ffs, N, "layer " + std::to_string(l) + "'s gate | up");
        shape(w.ffn_down, N, ffs, "layer " + std::to_string(l) + "'s down");
        st.done(l);
    }
    c.out_norm = up.f32(st.t(NL, "output_norm.weight"), N);
    const Tensor& out_t = st.t(NL, "output.weight");
    if ((int) out_t.ne[0] != N || (int64_t) out_t.ne[1] < c.v1)
        throw std::runtime_error("output.weight's rows: wrong shape");
    if (lmhead_6p2_enabled()) {
        if (!clusters_file.empty()) {
            CardClusterSlice cs;
            load_vhead_clusters_for_card(clusters_file, N, c.v0, c.v1, cs);
            c.out_6p2 = up.vhead_6p2(out_t, c.v0, c.v1, cs.perm_to_vocab.data(), cs.vocab_to_local.data());
            c.out_cent_f16 = up.put<uint16_t>(cs.cent_f16.data(), cs.cent_f16.size() * 2);
            c.out_cluster_off = up.put<int32_t>(cs.cluster_off.data(), cs.cluster_off.size() * 4);
            c.out_n_clusters = cs.n_clusters;
        } else {
            c.out_6p2 = up.vhead_6p2(out_t, c.v0, c.v1);
        }
    } else {
        c.out = up.dense(up.rows(out_t, range(c.v0, c.v1)));
        q8(c.out, "output.weight");
        shape(c.out, c.v1 - c.v0, N, "output.weight's rows");
    }
    st.done(NL);
    {
        auto T = [&](const char* s) -> const Tensor& { return st.t(NL + 1, "blk." + std::to_string(NL) + "." + s); };
        attn(c.mtp, T);
        ffn(c.mtp, T);
        c.mtp_eh = up.dense(up.rows(T("nextn.eh_proj.weight"), range(0, N)));   // every card the whole of it
        c.mtp_enorm = up.f32(T("nextn.enorm.weight"), N);
        c.mtp_hnorm = up.f32(T("nextn.hnorm.weight"), N);
        c.mtp_head_norm = up.f32(T("nextn.shared_head_norm.weight"), N);
        for (const DW* d : {&c.mtp.wq, &c.mtp.wo, &c.mtp.ffn_gu, &c.mtp.ffn_down, &c.mtp_eh}) q8(*d, "a draft layer matrix");
        shape(c.mtp.wq, qkv, N, "the draft layer's q | k | v");
        shape(c.mtp.wo, N, c.nq * HD, "the draft layer's attn_output");
        shape(c.mtp.ffn_gu, 2 * ffs, N, "the draft layer's gate | up");
        shape(c.mtp.ffn_down, N, ffs, "the draft layer's down");
        shape(c.mtp_eh, N, 2 * N, "the draft layer's eh_proj");
        st.done(NL + 1);
    }
}
#endif

// The tensors come from st: group l is layer l's, group NL the output head's (stage_groups); each group is given back
// once this card is through with it.
// host_experts: every routed expert to a host copy (an ExpertCache, or the lab's holder); store: the holder's shared memory
// cold_experts: non-null in FNX_CPU_EXPERTS=1 mode (single un-duplicated cold host buffer; resident experts straight to VRAM)
void load_card(Stager& st, CardW& c, int card_idx, const std::vector<std::vector<char>>& resident,
               const std::vector<double>* arena_score, Store* store, bool host_experts,
               std::vector<ColdExpertInfo>* cold_experts, const std::string& clusters_file = "") {
    ck(cudaSetDevice(c.dev), "cudaSetDevice");
    Up up{c, store};
    auto L = [&](int l, const char* s) -> const Tensor& { return st.t(l, "blk." + std::to_string(l) + "." + s); };
    // recurrent channel lists: q, k for this card's key heads; v for its value heads (s * NKH + k)
    std::vector<int> qch, kch, vch, vblocks;
    const std::vector<int>& vheads = c.vheads;
    for (int k = c.k0; k < c.k0 + c.nk; ++k)
        for (int i = 0; i < HS; ++i) { qch.push_back(k * HS + i); kch.push_back(NKH * HS + k * HS + i); }
    {
        std::vector<int> vmap;
        std::vector<bool> seen(c.nk, false);
        for (int h : vheads) {
            const int kl = h % NKH - c.k0;
            if (kl < 0 || kl >= c.nk) throw std::runtime_error("fnx: a value head whose key head is not on its card");
            vmap.push_back(kl | (seen[kl] ? 0 : 1 << 8));
            seen[kl] = true;
            for (int i = 0; i < HS; ++i) vch.push_back(2 * NKH * HS + h * HS + i);
            for (int b = 0; b < HS / 32; ++b) vblocks.push_back(h * HS / 32 + b);
        }
        for (int kl = 0; kl < c.nk; ++kl)
            if (!seen[kl]) throw std::runtime_error("fnx: a key head with no value head on its card");
        c.vmap = up.put<int>(vmap.data(), vmap.size() * 4);
    }
    std::vector<int> conv_rows = qch;
    conv_rows.insert(conv_rows.end(), kch.begin(), kch.end());
    conv_rows.insert(conv_rows.end(), vch.begin(), vch.end());
    std::vector<int> zrows;
    for (int ch : vch) zrows.push_back(ch - 2 * NKH * HS);
    // attention rows: q heads (q + gate, 2 * HD rows each), KV heads; wo input blocks of the q heads
    std::vector<int> qrows, kvrows, woblocks;
    for (int h = c.q0; h < c.q0 + c.nq; ++h) {
        for (int i = 0; i < 2 * HD; ++i) qrows.push_back(h * 2 * HD + i);
        for (int b = 0; b < HD / 32; ++b) woblocks.push_back(h * HD / 32 + b);
    }
    for (int h = c.kv0; h < c.kv0 + c.nkv; ++h)
        for (int i = 0; i < HD; ++i) kvrows.push_back(h * HD + i);
#if FNX_27B
    (void) card_idx, (void) resident, (void) arena_score, (void) host_experts, (void) cold_experts;
    load_layers_27b(st, c, up, conv_rows, zrows, vblocks, qrows, kvrows, woblocks, clusters_file);
    return;
#endif
    (void) clusters_file;
    // the K/V cache is split by position: every card projects q for every head and k/v for both KV heads
    std::vector<int> qrows_all, kvrows_all;
    for (int i = 0; i < NQH * 2 * HD; ++i) qrows_all.push_back(i);
    for (int i = 0; i < NKVH * HD; ++i) kvrows_all.push_back(i);
    c.L.resize(NL);
    const bool cpu_mode = cold_experts != nullptr;
    std::vector<const char*> hgu(host_experts && !cpu_mode ? (size_t) NL * NEXP : 0), hdn(hgu.size());   // legacy host copies
    std::vector<const char*> cpu_ep_gu(cpu_mode ? (size_t) NL * NEXP : 0, nullptr);
    std::vector<const char*> cpu_ep_dn(cpu_mode ? (size_t) NL * NEXP : 0, nullptr);
    std::vector<uint8_t> cpu_res_mask(cpu_mode ? (size_t) NL * NEXP : 0, 0);
    Up ue{c, store, &c.expert_mem};
    size_t cpu_exp_bytes = 0;
    for (int l = 0; l < NL; ++l) {
        LayerW& w = c.L[l];
        const char** hg = host_experts && !cpu_mode ? hgu.data() + (size_t) l * NEXP : nullptr;
        const char** hd = host_experts && !cpu_mode ? hdn.data() + (size_t) l * NEXP : nullptr;
        const char* hp[2] = {"hc_attn_", "hc_ffn_"};
        for (int h = 0; h < 2; ++h) {
            w.hc_norm[h] = up.f32(L(l, (std::string(hp[h]) + "norm.weight").c_str()), D);
            w.hc_down[h] = up.q8split(L(l, (std::string(hp[h]) + "down.weight").c_str()));
            w.hc_up[h] = up.q8split(L(l, (std::string(hp[h]) + "up.weight").c_str()));
            w.hc_inj[h] = up.q8split_any(L(l, (std::string(hp[h]) + "inject.weight").c_str()));
        }
        const Tensor& gi = L(l, "ffn_gate_inp.weight");
        const Tensor& gs = L(l, "ffn_gate_inp_shexp.weight");
        if (gi.ne[1] != NEXP || gs.ne[1] != 1)
            throw std::runtime_error("fnx: " + gi.name + " holds " + std::to_string(gi.ne[1]) + " rows and " + gs.name + " " +
                                     std::to_string(gs.ne[1]) + ", not the " + std::to_string(NEXP) +
                                     " gate rows and 1 shared row k_router_route reads");
        w.router = up.f32_cat(gi, gs, N);
        if (!is_attn(l)) {
            w.qkvz = up.dense(up.rows2(L(l, "attn_qkv.weight"), conv_rows, L(l, "attn_gate.weight"), zrows));
            w.ab = up.f32_rows2(L(l, "ssm_alpha.weight"), L(l, "ssm_beta.weight"), vheads, N);
            w.ssm_out = up.dense(up.cols(L(l, "ssm_out.weight"), vblocks));
            w.conv = up.f32_rows(L(l, "ssm_conv1d.weight"), conv_rows, 4);
            std::vector<float> a, dt;
            const size_t nh = (size_t) *std::max_element(vheads.begin(), vheads.end()) + 1;
            const float* af = f32_data(L(l, "ssm_a"), nh);
            const float* df = f32_data(L(l, "ssm_dt.bias"), nh);
            for (int h : vheads) { a.push_back(af[h]); dt.push_back(df[h]); }
            w.ssm_a = up.put<float>(a.data(), a.size() * 4);
            w.dt_bias = up.put<float>(dt.data(), dt.size() * 4);
            w.ssm_norm = up.f32(L(l, "ssm_norm.weight"), HS);
        } else {
            w.wq = up.dense(up.rows(L(l, "attn_q.weight"), qrows_all));
            w.wk = up.dense(up.rows(L(l, "attn_k.weight"), kvrows_all));
            w.wv = up.dense(up.rows(L(l, "attn_v.weight"), kvrows_all));
            w.wo = up.dense(up.cols(L(l, "attn_output.weight"), woblocks));
            w.q_norm = up.f32(L(l, "attn_q_norm.weight"), HD);
            w.k_norm = up.f32(L(l, "attn_k_norm.weight"), HD);
            const Tensor& iq = L(l, "indexer.q_proj.weight");
            const Tensor& ik = L(l, "indexer.k_proj.weight");
            if (iq.ne[1] != QSA_HEADS * QSA_KEY || iq.ne[0] != N || ik.ne[1] != QSA_KEY || ik.ne[0] != N)
                throw std::runtime_error("fnx: " + iq.name + " is " + std::to_string(iq.ne[0]) + " by " + std::to_string(iq.ne[1]) +
                                         " and " + ik.name + " is " + std::to_string(ik.ne[0]) + " by " + std::to_string(ik.ne[1]) +
                                         ", not the " + std::to_string(N) + " by " + std::to_string(QSA_HEADS * QSA_KEY) + " and " +
                                         std::to_string(N) + " by " + std::to_string(QSA_KEY) + " k_idx_proj reads");
            w.idx_wq = up.bf16(iq);
            w.idx_wk = up.bf16(ik);
            w.idx_qn = up.f32(L(l, "indexer.q_norm.weight"), QSA_KEY);
            w.idx_kn = up.f32(L(l, "indexer.k_norm.weight"), QSA_KEY);
        }
        if (cpu_mode) {
            // FNX_CPU_EXPERTS=1: resident experts upload straight to VRAM; cold experts write this card's slice into
            // the single un-duplicated cold_experts host buffer (zero duplicate host copies of resident experts!)
            const Tensor& gt = L(l, "ffn_gate_exps.weight");
            const Tensor& ut = L(l, "ffn_up_exps.weight");
            const Tensor& dt = L(l, "ffn_down_exps.weight");
            if (gt.type != ut.type || gt.ne[0] != ut.ne[0] || gt.ne[2] != ut.ne[2])
                throw std::runtime_error(gt.name + " and " + ut.name + " do not stack");
            Up::row_range(gt, c.ff0, c.ff1);
            Up::row_range(ut, c.ff0, c.ff1);
            const TypeInfo dti = block32(dt);
            const int b0 = c.ff0 / 32, b1 = c.ff1 / 32;
            const size_t d_src_row = dt.row_bytes();
            if (b0 < 0 || b1 <= b0 || (size_t) b1 * dti.bytes > d_src_row)
                throw std::runtime_error(dt.name + ": blocks [" + std::to_string(b0) + ", " + std::to_string(b1) + ") out of bounds");
            w.gu_e = DW{};
            w.gu_e.type = gt.type;
            w.gu_e.n_in = (int) gt.ne[0];
            w.gu_e.n_out = 2 * (c.ff1 - c.ff0);
            w.gu_e.row_bytes = gt.row_bytes();
            w.gu_e_stride = (size_t) w.gu_e.n_out * w.gu_e.row_bytes;
            w.down_e = DW{};
            w.down_e.type = dt.type;
            w.down_e.n_in = (b1 - b0) * dti.block;
            w.down_e.n_out = (int) dt.ne[1];
            w.down_e.row_bytes = (size_t) (b1 - b0) * dti.bytes;
            w.down_e_stride = (size_t) w.down_e.n_out * w.down_e.row_bytes;

            const size_t gu_half = (size_t) (c.ff1 - c.ff0) * w.gu_e.row_bytes;
            const size_t gu_src_mat = (size_t) gt.ne[1] * w.gu_e.row_bytes;
            const size_t dn_src_mat = (size_t) dt.ne[1] * d_src_row;
            int n_res = 0;
            for (int e = 0; e < NEXP; ++e) n_res += resident[(size_t) l][(size_t) e] ? 1 : 0;
            std::vector<uint8_t> tmp_gu((size_t) n_res * w.gu_e_stride);
            std::vector<uint8_t> tmp_dn((size_t) n_res * w.down_e_stride);
            int ri = 0;
            for (int e = 0; e < NEXP; ++e) {
                const size_t idx = (size_t) l * NEXP + (size_t) e;
                const uint8_t* g_src = gt.data + (size_t) e * gu_src_mat + (size_t) c.ff0 * w.gu_e.row_bytes;
                const uint8_t* u_src = ut.data + (size_t) e * gu_src_mat + (size_t) c.ff0 * w.gu_e.row_bytes;
                const uint8_t* d_src = dt.data + (size_t) e * dn_src_mat + (size_t) b0 * dti.bytes;
                uint8_t* g_dst = nullptr;
                uint8_t* d_dst = nullptr;
                if (resident[(size_t) l][(size_t) e]) {
                    g_dst = tmp_gu.data() + (size_t) ri * w.gu_e_stride;
                    d_dst = tmp_dn.data() + (size_t) ri * w.down_e_stride;
                    ++ri;
                } else {
                    ColdExpertInfo& ci = (*cold_experts)[idx];
                    g_dst = const_cast<uint8_t*>(ci.gu_half[card_idx]);
                    d_dst = const_cast<uint8_t*>(ci.dn_half[card_idx]);
                }
                std::memcpy(g_dst, g_src, gu_half);
                std::memcpy(g_dst + gu_half, u_src, gu_half);
                for (int r = 0; r < w.down_e.n_out; ++r)
                    std::memcpy(d_dst + (size_t) r * w.down_e.row_bytes, d_src + (size_t) r * d_src_row, w.down_e.row_bytes);
            }
            if (n_res > 0) {
                const size_t before = c.bytes;
                uint8_t* d_gu = ue.put<uint8_t>(tmp_gu.data(), tmp_gu.size());
                uint8_t* d_dn = ue.put<uint8_t>(tmp_dn.data(), tmp_dn.size());
                cpu_exp_bytes += c.bytes - before;
                int slot = 0;
                for (int e = 0; e < NEXP; ++e) {
                    const size_t idx = (size_t) l * NEXP + (size_t) e;
                    if (resident[(size_t) l][(size_t) e]) {
                        cpu_ep_gu[idx] = (const char*) (d_gu + (size_t) slot * w.gu_e_stride);
                        cpu_ep_dn[idx] = (const char*) (d_dn + (size_t) slot * w.down_e_stride);
                        cpu_res_mask[idx] = 1;
                        ++slot;
                    }
                }
            }
        } else {
            w.gu_e = up.expert_rows2(L(l, "ffn_gate_exps.weight"), L(l, "ffn_up_exps.weight"), c.ff0, c.ff1, w.gu_e_stride, hg);
            w.down_e = up.expert_cols(L(l, "ffn_down_exps.weight"), c.ff0 / 32, c.ff1 / 32, w.down_e_stride, hd);
        }
        w.sh_gu = up.dense(up.rows2(L(l, "ffn_gate_shexp.weight"), range(c.ff0, c.ff1), L(l, "ffn_up_shexp.weight"), range(c.ff0, c.ff1)));
        w.sh_down = up.dense(up.cols(L(l, "ffn_down_shexp.weight"), range(c.ff0 / 32, c.ff1 / 32)));
        if (!w.sh_gu.s.q || !w.sh_down.s.q) throw std::runtime_error("fnx: the shared expert is not Q8_0");
        if (l == PLE_LAYER) {
            const Tensor& pk = L(l, "ple_key.weight");
            const Tensor& pv = L(l, "ple_value.weight");
            // dense() reads N-wide xq rows and writes D-float rows into b->ple_key, N-float into b->ple_val
            if (pk.ne[0] != N || pk.ne[1] != D)
                throw std::runtime_error("fnx: " + pk.name + " is " + std::to_string(pk.ne[0]) + " by " + std::to_string(pk.ne[1]) +
                                         ", not the " + std::to_string(N) + " by " + std::to_string(D) + " its dense into the ple_key rows reads");
            if (pv.ne[0] != N || pv.ne[1] != N)
                throw std::runtime_error("fnx: " + pv.name + " is " + std::to_string(pv.ne[0]) + " by " + std::to_string(pv.ne[1]) +
                                         ", not the " + std::to_string(N) + " by " + std::to_string(N) + " its dense into the ple_val rows reads");
            c.ple_key = up.dense(up.rows(pk, range(0, (int) pk.ne[1])));
            c.ple_value = up.dense(up.rows(pv, range(0, (int) pv.ne[1])));
            c.ple_norm_key = up.f32(L(l, "ple_norm_key.weight"), D);
            c.ple_norm_query = up.f32(L(l, "ple_norm_query.weight"), D);
            c.ple_norm_conv = up.f32(L(l, "ple_norm_conv.weight"), D);
            const Tensor& pc = L(l, "ple_conv1d.weight");   // k_ple_conv reads w[ch * 4 + k]: 4 a row, D of them
            if (pc.ne[0] != 4 || pc.ne[1] != D)
                throw std::runtime_error("fnx: " + pc.name + " is " + std::to_string(pc.ne[0]) + " by " + std::to_string(pc.ne[1]) +
                                         ", not the 4 by " + std::to_string(D) + " k_ple_conv reads");
            c.ple_conv = up.f32(pc, 4 * D);
        }
        st.done(l);
    }
    c.head_norm = up.f32(st.t(NL, "output_hc_norm.weight"), D);
    c.head_down = up.q8split(st.t(NL, "output_hc_down.weight"));
    c.head_up = up.q8split(st.t(NL, "output_hc_up.weight"));
    {
        const Tensor& out_t = st.t(NL, "output.weight");
        if (lmhead_6p2_enabled() && out_t.type == T_Q8_0 && (out_t.ne[0] & 511) == 0)
            c.out_6p2 = up.vhead_6p2(out_t, c.v0, c.v1);
        else
            c.out = up.dense(up.rows(out_t, range(c.v0, c.v1)));
    }
    st.done(NL);
    if (cpu_mode) {
        const size_t before = c.bytes;
        c.ec = ExpertCache{};
        c.ec.ep_gu = ue.put<const char*>(cpu_ep_gu.data(), cpu_ep_gu.size() * sizeof(char*));
        c.ec.ep_dn = ue.put<const char*>(cpu_ep_dn.data(), cpu_ep_dn.size() * sizeof(char*));
        c.resident_mask = ue.put<uint8_t>(cpu_res_mask.data(), cpu_res_mask.size());
        c.expert_bytes = cpu_exp_bytes + (c.bytes - before);
        for (int l = 0; l < NL; ++l) {
            c.L[(size_t) l].gu_e.ep = c.ec.ep_gu + (size_t) l * NEXP;
            c.L[(size_t) l].down_e.ep = c.ec.ep_dn + (size_t) l * NEXP;
        }
    } else if (!resident.empty()) {   // tracked, so the engine can place them again with more in RAM (Engine: VRAM short)
        const size_t before = c.bytes;
        ue.expert_cache(resident, hgu, hdn, arena_score);
        c.expert_bytes = c.bytes - before;
    } else if (host_experts) c.ec_host_gu = hgu, c.ec_host_dn = hdn;   // the holder: placed by a worker
}

// ---------------------------------------------------------------- the draft layer
#if !FNX_27B
// Strata's runtime files: dense.txt lines "name kind rows cols offset bytes" into dense.bin (kind q8_0 | bf16 | f32)
struct MtpFiles {
    std::map<std::string, Tensor> t;
    void* base = nullptr;
    size_t size = 0;
    std::vector<uint8_t> gate_b, up_b, down_b;   // the experts' blocks: [512][640][2560], [512][640][2560], [512][2560][640]
    uint32_t etype = T_Q4_0;
    Tensor gate, up, down;
    explicit MtpFiles(const std::string& dir) {
        const int fd = ::open((dir + "/dense.bin").c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("mtp: cannot open " + dir + "/dense.bin");
        size = (size_t) lseek(fd, 0, SEEK_END);
        base = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        ::close(fd);
        if (base == MAP_FAILED) throw std::runtime_error("mtp: cannot map dense.bin");
        std::ifstream idx(dir + "/dense.txt");
        if (!idx) throw std::runtime_error("mtp: cannot open " + dir + "/dense.txt");
        std::string line;
        while (std::getline(idx, line)) {
            if (line.empty()) continue;
            std::istringstream is(line);
            std::string name, kind;
            int64_t rows, cols;
            uint64_t off, bytes;
            if (!(is >> name >> kind >> rows >> cols >> off >> bytes)) throw std::runtime_error("mtp: dense.txt: " + line);
            Tensor x;
            x.name = name;
            x.type = kind == "q8_0" ? T_Q8_0 : kind == "bf16" ? T_BF16 : kind == "f32" ? T_F32 : 0xffffffffu;
            if (x.type == 0xffffffffu) throw std::runtime_error("mtp: dense.txt: kind " + kind);
            x.ne[0] = cols;
            x.ne[1] = rows;
            x.n_dims = 2;
            x.data = (const uint8_t*) base + off;
            x.nbytes = bytes;
            if (off + bytes > size || x.row_bytes() * rows != bytes) throw std::runtime_error("mtp: dense.txt: " + name + " does not fit");
            t[name] = x;
        }
        // FNX_MTP_EXPERTS: q2_0 is Strata's experts.bin (unpacked to Q4_0 blocks), q4_0 is tools/mtp_experts.py's
        // experts-q4_0.bin, made from the BF16 checkpoint; the same VRAM either way. Q8_0 drafted exactly as Q4_0 did
        // for ~420 MiB more a card, which the arena's reserve does not count: not offered.
        const char* ex_s = std::getenv("FNX_MTP_EXPERTS");
        const std::string ex = ex_s ? ex_s : "q4_0";
        if (ex == "q2_0") experts(dir + "/experts.bin");
        else if (ex == "q4_0") experts_blocks(dir + "/experts-q4_0.bin", T_Q4_0, 18);
        else throw std::runtime_error("mtp: FNX_MTP_EXPERTS=" + ex + ": q2_0 or q4_0");
        std::fprintf(stderr, "fnx: draft layer experts %s\n", ex.c_str());
        views();
    }
    ~MtpFiles() { if (base) munmap(base, size); }
    const Tensor& operator[](const std::string& n) const {
        const auto it = t.find(n);
        if (it == t.end()) throw std::runtime_error("mtp: " + n + " is missing");
        return it->second;
    }
    // experts.bin: per expert codes (gate/up rows interleaved, 640 B each; then 2560 down rows of 160 B), then the
    // fp16 scales (gate/up rows, 40 each; down rows, 10 each). Q2_0: value j of a 64-value block is code j (byte j / 4,
    // bits 2 (j % 4)) minus one, times the scale. Q4_0 keeps 32 values per block: nibble = code + 7 (value i low,
    // i + 16 high), the same scale for both halves of a Q2_0 block.
    void experts(const std::string& path) {
        constexpr int H = N, F = FF, NE = NEXP;
        constexpr size_t ROW_GU = H / 4, ROW_D = F / 4, SC_GU = H / 64, SC_D = F / 64;
        constexpr size_t O_D = 2ull * F * ROW_GU, O_GS = O_D + (size_t) H * ROW_D, O_DS = O_GS + 2ull * F * SC_GU * 2;
        constexpr size_t BLOB = O_DS + (size_t) H * SC_D * 2;
#if !FNX_27B
        static_assert(BLOB == 1382400, "Strata's Q2_0 expert blob");
#endif
        std::FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) throw std::runtime_error("mtp: cannot open " + path);
        const size_t qrow_gu = (size_t) H / 32 * 18, qrow_d = (size_t) F / 32 * 18;
        gate_b.resize((size_t) NE * F * qrow_gu);
        up_b.resize((size_t) NE * F * qrow_gu);
        down_b.resize((size_t) NE * H * qrow_d);
        etype = T_Q4_0;
        std::vector<uint8_t> blob(BLOB);
        // one row of n values: codes (n / 4 bytes) and n / 64 scales -> n / 32 Q4_0 blocks
        auto row = [](const uint8_t* codes, const uint8_t* scales, int n, uint8_t* out) {
            for (int b = 0; b < n / 64; ++b) {
                uint16_t d;
                std::memcpy(&d, scales + b * 2, 2);
                const uint8_t* c = codes + b * 16;
                auto code = [&](int j) { return (c[j / 4] >> (2 * (j % 4))) & 3; };
                for (int h = 0; h < 2; ++h) {
                    uint8_t* o = out + (size_t) (2 * b + h) * 18;
                    std::memcpy(o, &d, 2);
                    for (int i = 0; i < 16; ++i)
                        o[2 + i] = (uint8_t) ((code(32 * h + i) + 7) | ((code(32 * h + 16 + i) + 7) << 4));
                }
            }
        };
        for (int e = 0; e < NE; ++e) {
            if (std::fread(blob.data(), 1, BLOB, f) != BLOB) {
                std::fclose(f);
                throw std::runtime_error("mtp: experts.bin is truncated");
            }
            for (int r = 0; r < F; ++r) {
                row(&blob[(2 * r) * ROW_GU], &blob[O_GS + (2 * r) * SC_GU * 2], H, &gate_b[((size_t) e * F + r) * qrow_gu]);
                row(&blob[(2 * r + 1) * ROW_GU], &blob[O_GS + (2 * r + 1) * SC_GU * 2], H, &up_b[((size_t) e * F + r) * qrow_gu]);
            }
            for (int r = 0; r < H; ++r)
                row(&blob[O_D + r * ROW_D], &blob[O_DS + r * SC_D * 2], F, &down_b[((size_t) e * H + r) * qrow_d]);
        }
        std::fclose(f);
    }
    // experts-q4_0.bin / experts-q8_0.bin: gate, up, down, each [512][rows] rows of 32-value blocks of `bb` bytes
    void experts_blocks(const std::string& path, uint32_t type, size_t bb) {
        const size_t qrow_gu = (size_t) N / 32 * bb, qrow_d = (size_t) FF / 32 * bb;
        gate_b.resize((size_t) NEXP * FF * qrow_gu);
        up_b.resize((size_t) NEXP * FF * qrow_gu);
        down_b.resize((size_t) NEXP * N * qrow_d);
        etype = type;
        std::FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) throw std::runtime_error("mtp: cannot open " + path + " (tools/mtp_experts.py makes it)");
        const bool ok = std::fread(gate_b.data(), 1, gate_b.size(), f) == gate_b.size() &&
                        std::fread(up_b.data(), 1, up_b.size(), f) == up_b.size() &&
                        std::fread(down_b.data(), 1, down_b.size(), f) == down_b.size() && std::fgetc(f) == EOF;
        std::fclose(f);
        if (!ok) throw std::runtime_error("mtp: " + path + " is not " + std::to_string(gate_b.size() * 2 + down_b.size()) + " bytes");
    }
    void views() {
        constexpr int H = N, F = FF;
        auto view = [this](const char* name, std::vector<uint8_t>& v, int64_t cols, int64_t rows) {
            Tensor x;
            x.name = name;
            x.type = etype;
            x.ne[0] = cols;
            x.ne[1] = rows;
            x.ne[2] = NEXP;
            x.n_dims = 3;
            x.data = v.data();
            x.nbytes = v.size();
            return x;
        };
        gate = view("mtp gate", gate_b, H, F);
        up = view("mtp up", up_b, H, F);
        down = view("mtp down", down_b, F, H);
    }
};
#endif

void free_mtp_cards(std::vector<MtpCard>& cards) {
    for (MtpCard& mc : cards) {
        if (mc.allocs.empty()) continue;
        if (cudaSetDevice(mc.dev) != cudaSuccess) std::fprintf(stderr, "mtp: cudaSetDevice(%d) failed while freeing\n", mc.dev);
        for (size_t i = 0; i < mc.allocs.size(); ++i) {
            const cudaError_t e = cudaFree(mc.allocs[i]);
            if (e == cudaSuccess) continue;
            cudaGetLastError();
            cudaPointerAttributes at{};
            const cudaError_t ea = cudaPointerGetAttributes(&at, mc.allocs[i]);
            cudaGetLastError();
            std::fprintf(stderr, "mtp: card %d: cudaFree of allocation %zu of %zu (%p) failed: %s (%s); the pointer: %s, type %d, device %d\n",
                         mc.dev, i, mc.allocs.size(), mc.allocs[i], cudaGetErrorName(e), cudaGetErrorString(e),
                         ea == cudaSuccess ? "known" : cudaGetErrorName(ea), (int) at.type, at.device);
        }
        mc.allocs.clear();
    }
}

#if !FNX_27B
void load_mtp_card(const Model& m, const MtpFiles& f, const CardW& c, const std::vector<int>& vocab, MtpCard& mc) {
    ck(cudaSetDevice(c.dev), "cudaSetDevice");
    CardW sink;   // Up counts bytes into a CardW
    sink.dev = c.dev;
    mc.dev = c.dev;
    Up up{sink, nullptr, &mc.allocs};
    auto bf16_of = [&](const Tensor& t) {
        if (t.type != T_BF16) throw std::runtime_error("mtp: " + t.name + " is not bf16");
        return (const uint16_t*) t.data;
    };
    auto q8 = [&](const char* name) { return up.q8split_bf16(bf16_of(f[name]), (size_t) f[name].ne[0] * f[name].ne[1]); };
    std::vector<int> qrows, kvrows, woblocks;
    for (int h = c.q0; h < c.q0 + c.nq; ++h) {
        for (int i = 0; i < 2 * HD; ++i) qrows.push_back(h * 2 * HD + i);
        for (int b = 0; b < HD / 32; ++b) woblocks.push_back(h * HD / 32 + b);
    }
    for (int h = c.kv0; h < c.kv0 + c.nkv; ++h)
        for (int i = 0; i < HD; ++i) kvrows.push_back(h * HD + i);
    // the K/V cache is split by position: every card projects q for every head and k/v for both KV heads
    std::vector<int> qrows_all, kvrows_all;
    for (int i = 0; i < NQH * 2 * HD; ++i) qrows_all.push_back(i);
    for (int i = 0; i < NKVH * HD; ++i) kvrows_all.push_back(i);
    LayerW& w = mc.L;
    const char* hp[2] = {"attn_hyper_connection.", "mlp_hyper_connection."};
    for (int h = 0; h < 2; ++h) {
        const std::string p = hp[h];
        w.hc_norm[h] = up.f32(f[p + "hc_norm.weight"], D);
        w.hc_down[h] = q8((p + "input_mix_weight_down.weight").c_str());
        w.hc_up[h] = q8((p + "input_mix_weight_up.weight").c_str());
        const Tensor& inj = f[p + "block_inject_weight.weight"];
        const size_t inj_n = (size_t) inj.ne[0] * inj.ne[1];
        if (inj_n != (size_t) HC * D)
            throw std::runtime_error("mtp: " + inj.name + " has " + std::to_string(inj_n) + " elements, expected " +
                                     std::to_string((size_t) HC * D));
        w.hc_inj[h] = up.q8split_bf16(bf16_of(inj), inj_n);
    }
    {   // the router's rows, then the shared expert's gate, as f32
        const Tensor &r = f["mlp.gate.weight"], &sg = f["mlp.shared_expert_gate.weight"];
        if (r.ne[1] != NEXP || r.ne[0] != N)
            throw std::runtime_error("mtp: mlp.gate.weight is " + std::to_string(r.ne[0]) + " by " + std::to_string(r.ne[1]) +
                                     ", not the " + std::to_string(N) + " by " + std::to_string(NEXP) + " the router reads");
        if (sg.ne[0] != N || sg.ne[1] != 1)
            throw std::runtime_error("mtp: mlp.shared_expert_gate.weight is " + std::to_string(sg.ne[0]) + " by " +
                                     std::to_string(sg.ne[1]) + ", not the " + std::to_string(N) + " by 1 it reads");
        std::vector<float> h((size_t) (NEXP + 1) * N);
        const uint16_t* a = bf16_of(r);
        for (size_t i = 0; i < (size_t) NEXP * N; ++i) { const uint32_t u = (uint32_t) a[i] << 16; std::memcpy(&h[i], &u, 4); }
        const uint16_t* b = bf16_of(sg);
        for (size_t i = 0; i < (size_t) N; ++i) { const uint32_t u = (uint32_t) b[i] << 16; std::memcpy(&h[(size_t) NEXP * N + i], &u, 4); }
        w.router = up.put<float>(h.data(), h.size() * 4);
    }
    w.wq = up.dense(up.rows(f["self_attn.q_proj.weight"], qrows_all));
    w.wk = up.dense(up.rows(f["self_attn.k_proj.weight"], kvrows_all));
    w.wv = up.dense(up.rows(f["self_attn.v_proj.weight"], kvrows_all));
    w.wo = up.dense(up.cols(f["self_attn.o_proj.weight"], woblocks));
    w.q_norm = up.f32(f["self_attn.q_norm.weight"], HD);
    w.k_norm = up.f32(f["self_attn.k_norm.weight"], HD);
    const Tensor& iqk = f["self_attn.indexer.index_qk_proj.weight"];
    if (iqk.ne[1] != (QSA_HEADS + 1) * QSA_KEY || iqk.ne[0] != N)
        throw std::runtime_error("mtp: " + iqk.name + " is " + std::to_string(iqk.ne[0]) + " by " + std::to_string(iqk.ne[1]) + ", not the " +
                                 std::to_string(N) + " by " + std::to_string((QSA_HEADS + 1) * QSA_KEY) + " (queries then key) it reads");
    w.idx_qk = up.dense(up.rows(iqk, range(0, (QSA_HEADS + 1) * QSA_KEY)));
    w.idx_qn = up.f32(f["self_attn.indexer.q_layernorm.weight"], QSA_KEY);
    w.idx_kn = up.f32(f["self_attn.indexer.k_layernorm.weight"], QSA_KEY);
    w.gu_e = up.expert_rows2(f.gate, f.up, c.ff0, c.ff1, w.gu_e_stride);
    w.down_e = up.expert_cols(f.down, c.ff0 / 32, c.ff1 / 32, w.down_e_stride);
    w.sh_gu = up.dense(up.rows2(f["mlp.shared_expert.gate_proj.weight"], range(c.ff0, c.ff1), f["mlp.shared_expert.up_proj.weight"],
                                range(c.ff0, c.ff1)));
    w.sh_down = up.dense(up.cols(f["mlp.shared_expert.down_proj.weight"], range(c.ff0 / 32, c.ff1 / 32)));
    mc.fc_emb = up.dense(up.rows(f["fc_embedding.weight"], range(0, N)));
    mc.fc_hid = up.dense(up.rows(f["fc_hidden.weight"], range(0, N)));
    mc.norm_emb = up.f32(f["pre_fc_norm_embedding.weight"], N);
    mc.norm_hid = up.f32(f["pre_fc_norm_hidden.weight"], D);
    mc.mix_norm = up.f32(f["hyper_connection_mixer.hc_norm.weight"], D);
    mc.mix_down = q8("hyper_connection_mixer.input_mix_weight_down.weight");
    mc.mix_up = q8("hyper_connection_mixer.input_mix_weight_up.weight");
    std::vector<int> rows(vocab.begin() + mc.dv0, vocab.begin() + mc.dv1);
    mc.head = up.dense(up.rows(*m.g.find("output.weight"), rows));
    for (const DW* d : {&mc.head, &w.wq, &w.wk, &w.wv, &w.wo, &w.idx_qk, &w.sh_gu, &w.sh_down, &mc.fc_emb, &mc.fc_hid})
        if (!d->s.q) throw std::runtime_error("mtp: a projection is not Q8_0");
}
#endif

void read_draft_vocab(const std::string& vf, int model_vocab, std::vector<int>& vocab) {
    vocab.clear();
    std::FILE* f = std::fopen(vf.c_str(), "rb");
    if (!f) throw std::runtime_error("mtp: cannot open " + vf);
    struct stat stb{};
    if (fstat(fileno(f), &stb) != 0 || stb.st_size % 4 != 0) {
        std::fclose(f);
        throw std::runtime_error("mtp: " + vf + " is not whole int32 token ids (" + std::to_string((long long) stb.st_size) + " bytes)");
    }
    int32_t id;
    while (std::fread(&id, 4, 1, f) == 1) vocab.push_back(id);
    const bool io_bad = std::ferror(f) != 0;
    std::fclose(f);
    if (io_bad || vocab.size() * 4 != (size_t) stb.st_size)
        throw std::runtime_error("mtp: " + vf + " stopped at " + std::to_string(vocab.size() * 4) + " of its " +
                                 std::to_string((long long) stb.st_size) + " bytes");
    if (vocab.empty()) throw std::runtime_error("mtp: " + vf + " is empty");
    for (int id2 : vocab)
        if (id2 < 0 || id2 >= model_vocab) throw std::runtime_error("mtp: " + vf + " holds a token outside the vocabulary");
}

#if !FNX_27B
bool ends_with(const std::string& s, const char* suf) {
    const size_t n = std::strlen(suf);
    return s.size() >= n && s.compare(s.size() - n, n, suf) == 0;
}

std::string resolve_mtp_gguf_path(const std::string& dir) {
    struct stat st{};
    if (!dir.empty()) {
        if (ends_with(dir, ".gguf") || (stat(dir.c_str(), &st) == 0 && S_ISREG(st.st_mode))) return dir;
    }
    const char* env_g = std::getenv("FNX_MTP_GGUF");
    if (env_g && *env_g) {
        if (dir.empty() || dir == "gguf" || access((dir + "/dense.bin").c_str(), R_OK) != 0) return env_g;
    }
    return "";
}

void load_mtp_gguf_card(const Model& m, const Gguf& mg, int mtp_blk, const CardW& c, const std::vector<int>& vocab, MtpCard& mc,
                        size_t& out_bytes) {
    ck(cudaSetDevice(c.dev), "cudaSetDevice");
    CardW sink;   // Up counts bytes into a CardW
    sink.dev = c.dev;
    mc.dev = c.dev;
    Up up{sink, nullptr, &mc.allocs};
    const std::string bp = "blk." + std::to_string(mtp_blk) + ".";
    auto T = [&](const char* s) -> const Tensor& { return mg.t(bp + s); };
    std::vector<int> woblocks, qrows_all, kvrows_all;
    for (int h = c.q0; h < c.q0 + c.nq; ++h)
        for (int b = 0; b < HD / 32; ++b) woblocks.push_back(h * HD / 32 + b);
    for (int i = 0; i < NQH * 2 * HD; ++i) qrows_all.push_back(i);
    for (int i = 0; i < NKVH * HD; ++i) kvrows_all.push_back(i);

    LayerW& w = mc.L;
    const char* hp[2] = {"hc_attn_", "hc_ffn_"};
    for (int h = 0; h < 2; ++h) {
        w.hc_norm[h] = up.f32(T((std::string(hp[h]) + "norm.weight").c_str()), D);
        w.hc_down[h] = up.q8split_any(T((std::string(hp[h]) + "down.weight").c_str()));
        w.hc_up[h] = up.q8split_any(T((std::string(hp[h]) + "up.weight").c_str()));
        w.hc_inj[h] = up.q8split_any(T((std::string(hp[h]) + "inject.weight").c_str()));
    }
    {
        const Tensor& gi = T("ffn_gate_inp.weight");
        const Tensor& gs = T("ffn_gate_inp_shexp.weight");
        if (gi.ne[0] != N || gi.ne[1] != NEXP || gs.ne[0] != N || gs.ne[1] != 1)
            throw std::runtime_error("mtp gguf: router tensors have unexpected shape");
        if (gi.type == T_F32 && gs.type == T_F32) {
            w.router = up.f32_cat(gi, gs, N);
        } else if (gi.type == T_BF16 && gs.type == T_BF16) {
            std::vector<float> h((size_t) (NEXP + 1) * N);
            const uint16_t* a = (const uint16_t*) gi.data;
            for (size_t i = 0; i < (size_t) NEXP * N; ++i) { const uint32_t u = (uint32_t) a[i] << 16; std::memcpy(&h[i], &u, 4); }
            const uint16_t* b = (const uint16_t*) gs.data;
            for (size_t i = 0; i < (size_t) N; ++i) { const uint32_t u = (uint32_t) b[i] << 16; std::memcpy(&h[(size_t) NEXP * N + i], &u, 4); }
            w.router = up.put<float>(h.data(), h.size() * 4);
        } else {
            throw std::runtime_error("mtp gguf: router tensors must be F32 or BF16");
        }
    }
    w.wq = up.dense(up.rows(T("attn_q.weight"), qrows_all));
    w.wk = up.dense(up.rows(T("attn_k.weight"), kvrows_all));
    w.wv = up.dense(up.rows(T("attn_v.weight"), kvrows_all));
    w.wo = up.dense(up.cols(T("attn_output.weight"), woblocks));
    w.q_norm = up.f32(T("attn_q_norm.weight"), HD);
    w.k_norm = up.f32(T("attn_k_norm.weight"), HD);
    {
        const Tensor& iq = T("indexer.q_proj.weight");
        const Tensor& ik = T("indexer.k_proj.weight");
        if (iq.ne[1] != QSA_HEADS * QSA_KEY || iq.ne[0] != N || ik.ne[1] != QSA_KEY || ik.ne[0] != N)
            throw std::runtime_error("mtp gguf: indexer projection shape mismatch");
        w.idx_wq = up.bf16(iq);
        w.idx_wk = up.bf16(ik);
        w.idx_qn = up.f32(T("indexer.q_norm.weight"), QSA_KEY);
        w.idx_kn = up.f32(T("indexer.k_norm.weight"), QSA_KEY);
    }
    w.gu_e = up.expert_rows2(T("ffn_gate_exps.weight"), T("ffn_up_exps.weight"), c.ff0, c.ff1, w.gu_e_stride);
    w.down_e = up.expert_cols(T("ffn_down_exps.weight"), c.ff0 / 32, c.ff1 / 32, w.down_e_stride);
    w.sh_gu = up.dense(up.rows2(T("ffn_gate_shexp.weight"), range(c.ff0, c.ff1), T("ffn_up_shexp.weight"), range(c.ff0, c.ff1)));
    w.sh_down = up.dense(up.cols(T("ffn_down_shexp.weight"), range(c.ff0 / 32, c.ff1 / 32)));
    {
        // nextn.eh_proj.weight is [2*N, N] Q8_0: columns [0, N) = fc_embedding, columns [N, 2*N) = fc_hidden
        const Tensor& eh = T("nextn.eh_proj.weight");
        if (eh.ne[0] != 2 * N || eh.ne[1] != N)
            throw std::runtime_error("mtp gguf: " + eh.name + " is " + std::to_string(eh.ne[0]) + " by " + std::to_string(eh.ne[1]) +
                                     ", expected " + std::to_string(2 * N) + " by " + std::to_string(N));
        mc.fc_emb = up.dense(up.cols(eh, range(0, N / 32)));
        mc.fc_hid = up.dense(up.cols(eh, range(N / 32, 2 * N / 32)));
    }
    mc.norm_emb = up.f32(T("nextn.enorm.weight"), N);
    mc.norm_hid = up.f32(T("nextn.hnorm.weight"), D);
    mc.mix_norm = up.f32(T("nextn.hc_head_norm.weight"), D);
    mc.mix_down = up.q8split_any(T("nextn.hc_head_down.weight"));
    mc.mix_up = up.q8split_any(T("nextn.hc_head_up.weight"));
    std::vector<int> rows(vocab.begin() + mc.dv0, vocab.begin() + mc.dv1);
    mc.head = up.dense(up.rows(m.g.t("output.weight"), rows));
    for (const DW* d : {&mc.head, &w.wq, &w.wk, &w.wv, &w.wo, &w.sh_gu, &w.sh_down, &mc.fc_emb, &mc.fc_hid})
        if (!d->s.q) throw std::runtime_error("mtp gguf: a projection is not Q8_0");
    out_bytes = sink.bytes;
}
#endif

// every card's error, joined: the first is often only the abort another card's failure caused ("card 0: card 1 failed")
void throw_card_errors(const std::vector<std::string>& errs) {
    std::string all;
    for (size_t c = 0; c < errs.size(); ++c)
        if (!errs[c].empty()) all += (all.empty() ? "" : "; ") + ("card " + std::to_string(c) + ": " + errs[c]);
    if (!all.empty()) throw std::runtime_error(all);
}

// Estimate non-routed GPU weight bytes per card before load_card uploads them, for live VRAM residency budgeting.
size_t estimate_non_routed_per_card(const Gguf& g) {
    size_t copied = 0, split = 0;
    for (const auto& kv : g.tensors()) {
        const std::string& n = kv.first;
        const size_t nb = kv.second.nbytes;
        if (n == "token_embd.weight" || n == "per_layer_token_embd.weight") continue;
        if (n.find("_exps.weight") != std::string::npos) continue;
        const bool is_split =
            n.find("attn_qkv.") != std::string::npos || n.find("attn_gate.") != std::string::npos ||
            n.find("ssm_out.") != std::string::npos || n.find("ssm_alpha.") != std::string::npos ||
            n.find("ssm_beta.") != std::string::npos || n.find("ssm_conv1d.") != std::string::npos ||
            n.find("ssm_a") != std::string::npos || n.find("ssm_dt.") != std::string::npos ||
            n.find("attn_output.") != std::string::npos || n.find("_shexp.") != std::string::npos ||
            n == "output.weight";
        if (is_split) {
            size_t card_nb = (nb + NCARD - 1) / NCARD;
            if (n == "output.weight" && lmhead_6p2_enabled()) card_nb += card_nb / 16;   // 6+2 row_meta margin
            split += card_nb;
        } else {
            copied += nb;
        }
    }
    return copied + split;
}

// Read either sfnx's data/expert-routes.bin (2*NL*NEXP uint32 tallies) or PolyStrata's STRP v1 .profile
// (24-byte header + uint16 (layer, expert) pairs ranked hottest-first). Populates score[NL*NEXP] (higher = hotter).
void load_expert_profile(const std::string& prof, const std::vector<double>& bytes, std::vector<double>& score) {
    std::FILE* f = std::fopen(prof.c_str(), "rb");
    if (!f) throw std::runtime_error("fnx: FNX_EXPERT_PROFILE cannot open " + prof);
    char magic[4] = {};
    const bool got_magic = std::fread(magic, 1, 4, f) == 4;
    if (got_magic && std::memcmp(magic, "STRP", 4) == 0) {
        uint32_t hdr[5] = {};
        if (std::fread(hdr, sizeof(uint32_t), 5, f) != 5) {
            std::fclose(f);
            throw std::runtime_error("fnx: FNX_EXPERT_PROFILE " + prof + ": truncated STRP header");
        }
        const uint32_t ver = hdr[0], nl = hdr[1], ne = hdr[2], slots = hdr[3], n_ranked = hdr[4];
        if (ver != 1 || nl != (uint32_t) NL || ne != (uint32_t) NEXP || n_ranked > slots) {
            std::fclose(f);
            throw std::runtime_error("fnx: FNX_EXPERT_PROFILE " + prof + ": incompatible STRP profile (v" + std::to_string(ver) +
                                     " " + std::to_string(nl) + "x" + std::to_string(ne) + ", want v1 " + std::to_string(NL) + "x" +
                                     std::to_string(NEXP) + ")");
        }
        std::vector<uint16_t> raw((size_t) n_ranked * 2);
        if (n_ranked > 0 && std::fread(raw.data(), sizeof(uint16_t), raw.size(), f) != raw.size()) {
            std::fclose(f);
            throw std::runtime_error("fnx: FNX_EXPERT_PROFILE " + prof + ": truncated STRP ranked list");
        }
        std::fclose(f);
        score.assign((size_t) NL * NEXP, 0.0);
        for (uint32_t r = 0; r < n_ranked; ++r) {
            const uint32_t l = raw[(size_t) r * 2], e = raw[(size_t) r * 2 + 1];
            if (l >= (uint32_t) NL || e >= (uint32_t) NEXP)
                throw std::runtime_error("fnx: FNX_EXPERT_PROFILE " + prof + ": STRP entry out of range");
            const size_t idx = (size_t) l * NEXP + e;
            if (score[idx] == 0.0) score[idx] = (double) ((size_t) NL * NEXP - r);
        }
        return;
    }
    std::rewind(f);
    std::vector<unsigned> cnt((size_t) 2 * NL * NEXP);
    const bool ok = std::fread(cnt.data(), sizeof(unsigned), cnt.size(), f) == cnt.size() && std::fgetc(f) == EOF;
    std::fclose(f);
    if (!ok)
        throw std::runtime_error("fnx: FNX_EXPERT_PROFILE " + prof + ": not " + std::to_string(cnt.size()) +
                                 " tallies (or STRP profile)");
    // decode and prompt picks weighted equally (each kind's share of its total), per byte; higher is hotter
    double td = 0, tp = 0;
    for (int i = 0; i < NL * NEXP; ++i) { td += cnt[(size_t) i]; tp += cnt[(size_t) (NL * NEXP + i)]; }
    score.assign((size_t) NL * NEXP, 0.0);
    for (int l = 0; l < NL; ++l)
        for (int e = 0; e < NEXP; ++e) {
            const size_t i = (size_t) l * NEXP + e;
            score[i] = (cnt[i] / std::max(1.0, td) + cnt[(size_t) NL * NEXP + i] / std::max(1.0, tp)) / bytes[(size_t) l];
        }
}

void release_cold_host_allocs(Model& m) {
    for (size_t i = 0; i < m.cold_host_allocs.size(); ++i) {
        void* p = m.cold_host_allocs[i];
        if (!p) continue;
        if (m.cold_host_via_mmap) {
            cudaHostUnregister(p);
            cudaGetLastError();
            munmap(p, m.cold_host_alloc_sizes[i]);
        } else {
            cudaFreeHost(p);
            cudaGetLastError();
        }
    }
    m.cold_host_allocs.clear();
    m.cold_host_alloc_sizes.clear();
    m.cold_experts.clear();
    m.cold_host_bytes = 0;
}
}  // namespace

#if FNX_27B
// the 27B's draft head: this card's rows of output.weight for its share of the draft vocabulary (the layer itself came
// with the model). dir names where the layer comes from, so a Flash-Next runtime directory in a 27B config fails here
Mtp::Mtp(const Model& m, const std::string& dir, const std::string& vocab_file) {
    if (dir != "nextn") throw std::runtime_error("mtp: the 27B's draft layer is the GGUF's own: its directory is \"nextn\", not \"" + dir + "\"");
    const int fast_k = mtp_fasthead_clusters();
    if (fast_k > 0) {
        if (m.cards[0].out_6p2.hi && m.cards[0].out_cent_f16) {
            shared_6p2 = true;
            top_clusters = std::min(fast_k, m.cards[0].out_n_clusters);
            cards.resize(NCARD);
            for (int c = 0; c < NCARD; ++c) {
                cards[c].dev = m.cards[c].dev;
                cards[c].dv0 = m.cards[c].v0;
                cards[c].dv1 = m.cards[c].v1;
            }
            return;
        }
        if (std::getenv("FNX_MTP_FASTHEAD"))
            throw std::runtime_error("FNX_MTP_FASTHEAD requires FNX_LMHEAD_6P2=1 and VHead clusters");
    }
    if (vocab_file.empty()) throw std::runtime_error("mtp: the 27B needs a draft vocabulary file");
    read_draft_vocab(vocab_file, m.vocab, vocab);
    cards.resize(NCARD);
    const int step = ((int) vocab.size() + NCARD - 1) / NCARD;
    std::vector<std::thread> th;
    std::vector<std::string> errs(NCARD);
    for (int c = 0; c < NCARD; ++c) {
        cards[c].dv0 = std::min((int) vocab.size(), c * step);
        cards[c].dv1 = std::min((int) vocab.size(), (c + 1) * step);
        th.emplace_back([&, c] {
            try {
                MtpCard& mc = cards[c];
                ck(cudaSetDevice(m.cards[c].dev), "cudaSetDevice");
                CardW sink;   // Up counts bytes into a CardW
                sink.dev = mc.dev = m.cards[c].dev;
                Up up{sink, nullptr, &mc.allocs};
                mc.head = up.dense(up.rows(*m.g.find("output.weight"), std::vector<int>(vocab.begin() + mc.dv0, vocab.begin() + mc.dv1)));
                if (!mc.head.s.q) throw std::runtime_error("mtp: output.weight is not Q8_0");
            } catch (const std::exception& e) { errs[c] = e.what(); }
        });
    }
    for (auto& t : th) t.join();
    for (int c = 0; c < NCARD; ++c)
        if (!errs[c].empty()) {
            free_mtp_cards(cards);
            throw std::runtime_error("mtp card " + std::to_string(c) + ": " + errs[c]);
        }
}
#else
Mtp::Mtp(const Model& m, const std::string& dir, const std::string& vocab_file) {
    const std::string gguf_path = resolve_mtp_gguf_path(dir);
    if (!gguf_path.empty()) {
        std::string vf = vocab_file;
        if (vf.empty()) {
            if (const char* ev = std::getenv("FNX_MTP_VOCAB"); ev && *ev) vf = ev;
        }
        if (vf.empty()) {
            const size_t slash = gguf_path.find_last_of('/');
            const std::string beside = (slash != std::string::npos ? gguf_path.substr(0, slash + 1) : "") + "draft_vocab.bin";
            const std::string cands[] = {
                beside,
                std::string(FNX_DIR) + "/data/draft_vocab_en.bin",
                "/mnt/storage_8tb_linux/failsafe_runtime/PolyStrata-data/mtp/rt/draft_vocab.bin",
            };
            for (const std::string& cand : cands)
                if (access(cand.c_str(), R_OK) == 0) { vf = cand; break; }
        }
        if (!vf.empty()) {
            read_draft_vocab(vf, m.vocab, vocab);
        } else {
            vocab.resize((size_t) m.vocab);
            for (int i = 0; i < m.vocab; ++i) vocab[(size_t) i] = i;
        }
        const Gguf mg(gguf_path, /*apply_env=*/false);
        const int mtp_blk = mg.find("blk." + std::to_string(NL) + ".attn_q.weight") ? NL
                          : mg.find("blk.0.attn_q.weight") ? 0 : -1;
        if (mtp_blk < 0)
            throw std::runtime_error("mtp gguf: " + gguf_path + " has neither blk." + std::to_string(NL) + ".attn_q.weight nor blk.0.attn_q.weight");
        cards.resize(NCARD);
        const int step = ((int) vocab.size() + NCARD - 1) / NCARD;
        for (int c = 0; c < NCARD; ++c) {
            cards[c].dv0 = std::min((int) vocab.size(), c * step);
            cards[c].dv1 = std::min((int) vocab.size(), (c + 1) * step);
        }
        std::vector<std::thread> th;
        std::vector<std::string> errs(NCARD);
        std::vector<size_t> card_bytes(NCARD, 0);
        for (int c = 0; c < NCARD; ++c)
            th.emplace_back([&, c] {
                try {
                    load_mtp_gguf_card(m, mg, mtp_blk, m.cards[c], vocab, cards[c], card_bytes[(size_t) c]);
                } catch (const std::exception& e) { errs[c] = e.what(); }
            });
        for (auto& t : th) t.join();
        for (int c = 0; c < NCARD; ++c)
            if (!errs[c].empty()) {
                free_mtp_cards(cards);
                throw std::runtime_error("mtp card " + std::to_string(c) + ": " + errs[c]);
            }
        std::fprintf(stderr, "fnx: draft layer loaded from GGUF %s (blk.%d, experts type=%u, %zu draft tokens, %.1f MiB/card)\n",
                     gguf_path.c_str(), mtp_blk, cards[0].L.gu_e.type, vocab.size(), card_bytes[0] / (double) (1ull << 20));
        return;
    }
    {
        const std::string vf = vocab_file.empty() ? dir + "/draft_vocab.bin" : vocab_file;
        read_draft_vocab(vf, m.vocab, vocab);
    }
    const MtpFiles files(dir);
    cards.resize(NCARD);
    const int step = ((int) vocab.size() + NCARD - 1) / NCARD;
    for (int c = 0; c < NCARD; ++c) {
        cards[c].dv0 = std::min((int) vocab.size(), c * step);
        cards[c].dv1 = std::min((int) vocab.size(), (c + 1) * step);
    }
    std::vector<std::thread> th;
    std::vector<std::string> errs(NCARD);
    for (int c = 0; c < NCARD; ++c)
        th.emplace_back([&, c] {
            try { load_mtp_card(m, files, m.cards[c], vocab, cards[c]); } catch (const std::exception& e) { errs[c] = e.what(); }
        });
    for (auto& t : th) t.join();
    for (int c = 0; c < NCARD; ++c)
        if (!errs[c].empty()) {
            free_mtp_cards(cards);   // no destructor runs for a constructor that throws
            throw std::runtime_error("mtp card " + std::to_string(c) + ": " + errs[c]);
        }
}
#endif

Mtp::~Mtp() { free_mtp_cards(cards); }

Model::~Model() { release_cold_host_allocs(*this); }

Model::Model(const std::string& path, Load how, Store* store, const std::string* image)
    : g(path), has_engram_override(g.has_engram_override()), has_hybrid_splice(g.has_hybrid_splice()) {
    // the cards first: reading the embeddings into RAM below takes seconds on a cold cache
    {
        int ndev = 0;
        ck(cudaGetDeviceCount(&ndev), "cudaGetDeviceCount");
        // a build for another count than the cards it is given is a config or build mistake (a 2-card binary on
        // prod's three would serve, slower): CUDA_VISIBLE_DEVICES (the config's "gpu") must name exactly NCARD
        if (ndev != NCARD)
            throw std::runtime_error("fnx: built for " + std::to_string(NCARD) + " cards (FNX_NCARD), " + std::to_string(ndev) +
                                     " visible: set CUDA_VISIBLE_DEVICES (a config's \"gpu\") to " + std::to_string(NCARD) +
                                     " of them, or build for " + std::to_string(ndev) + " (make FNX_NCARD=" + std::to_string(ndev) + ")");
        check_cards();
    }
#if FNX_27B
    const std::string p = "qwen35.";
    // block_count holds the draft (nextn) layer too
    if (g.i(p + "block_count") != NL + g.i(p + "nextn_predict_layers") || g.i(p + "nextn_predict_layers") != 1 ||
        g.i(p + "embedding_length") != N || g.i(p + "feed_forward_length") != FFD || g.i(p + "ssm.group_count") != NKH ||
        g.i(p + "ssm.time_step_rank") != NVH || g.i(p + "attention.head_count") != NQH ||
        g.i(p + "attention.head_count_kv") != NKVH || g.i(p + "full_attention_interval") != 4 ||
        g.i(p + "attention.key_length") != HD || g.i(p + "attention.value_length") != HD || g.i(p + "ssm.state_size") != HS ||
        g.i(p + "ssm.inner_size") != NVH * HS || g.i(p + "ssm.conv_kernel") != 4 || g.i(p + "rope.dimension_count") != NROT)
        throw std::runtime_error("27b: this GGUF's geometry is not Qwen3.8-27B's");
#else
    const std::string p = "qwen4exp.";
    if (g.i(p + "block_count") != NL || g.i(p + "embedding_length") != N || g.i(p + "expert_count") != NEXP ||
        g.i(p + "ssm.group_count") != NKH || g.i(p + "ssm.time_step_rank") != NVH || g.i(p + "attention.head_count") != NQH ||
        g.i(p + "attention.head_count_kv") != NKVH || g.i(p + "hyper_connection.count") != HC ||
        g.i(p + "hyper_connection.low_rank") != LR || g.i(p + "expert_feed_forward_length") != FF ||
        g.i(p + "attention.indexer.head_count") != QSA_HEADS || g.i(p + "attention.indexer.key_length") != QSA_KEY ||
        g.i(p + "attention.indexer.top_k") != QSA_TOP_K || g.i(p + "expert_used_count") != TOPK ||
        g.i(p + "rope.dimension_count") != NROT)
        throw std::runtime_error("fnx: this GGUF's geometry is not Flash-Next's");
    // the shared expert is split by the routed experts' ff ranges (CardW ff0, ff1)
    for (int l = 0; l < NL; ++l) {
        const std::string b = "blk." + std::to_string(l) + ".";
        if (g.t(b + "ffn_gate_shexp.weight").ne[1] != FF || g.t(b + "ffn_up_shexp.weight").ne[1] != FF || g.t(b + "ffn_down_shexp.weight").ne[0] != FF)
            throw std::runtime_error("fnx: layer " + std::to_string(l) + "'s shared expert is not " + std::to_string(FF) + " wide");
    }
#endif
    eps = (float) g.f(p + "attention.layer_norm_rms_epsilon");
    rope_base = (float) g.f(p + "rope.freq_base");
    // the trained length bounds the context (Engine refuses past it, or past it x FNX_ROPE_SCALE), so a model must name it
    {
        const int64_t n = g.i(p + "context_length");
        if (n <= 0 || n > INT32_MAX) throw std::runtime_error("fnx: " + p + "context_length " + std::to_string(n) + " is not a context length");
        ctx_train = (int) n;
    }
    // the model's own rotary scaling, which Engine runs by default (choose_rope); what sFNX cannot follow throws here
    {
        const ModelRope mr = model_rope(g, p);
        yarn_factor = mr.yarn_factor;
        yarn_orig = mr.yarn_orig;
    }
    tok_embd = &g.t("token_embd.weight");
#if FNX_27B
    if (tok_embd->type != T_Q8_0) throw std::runtime_error("27b: token_embd.weight is not Q8_0 (the host's embedding reads Q8_0)");
    for (const char* v : {"FNX_NGRAM", "FNX_NGRAM_CACHE_MIB", "FNX_NGRAM_THREADS"})
        if (std::getenv(v)) throw std::runtime_error(std::string("27b: ") + v + " is set, and the 27B has no PLE table");
    vocab = (int) tok_embd->ne[1];
    if (g.t("output.weight").ne[1] != vocab || g.t("output.weight").ne[0] != N)
        throw std::runtime_error("27b: output.weight is not [" + std::to_string(N) + "][" + std::to_string(vocab) + "] (token_embd's vocabulary)");
    {   // the host gathers its rows every window: a row read from the disk costs milliseconds, so keep it in RAM
        const auto t0 = std::chrono::steady_clock::now();
        lock_in_ram(*tok_embd);
        std::fprintf(stderr, "27b: %s (%.1f GB) locked in RAM in %.1f s\n", tok_embd->name.c_str(), tok_embd->nbytes / 1e9,
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
#else
    ple_table = &g.t("per_layer_token_embd.weight");
    if (tok_embd->type != T_Q8_0) throw std::runtime_error("fnx: token_embd.weight is not Q8_0 (the host's embedding reads Q8_0)");
    if (ple_table->type != T_IQ4_NL && ple_table->type != T_Q8_0)
        throw std::runtime_error("fnx: per_layer_token_embd.weight is neither IQ4_NL nor Q8_0 (the host's PLE rows read those)");
    vocab = (int) tok_embd->ne[1];
    // the host gathers rows of these two every window: a row faulted in from the disk through the mapping costs
    // milliseconds, so they are locked in RAM; FNX_NGRAM=disk leaves the table on the disk and reads its rows (PleRows)
    {
        const PleRows::Mode mode = PleRows::mode_from_env();
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<const Tensor*> lock{tok_embd};
        if (mode == PleRows::Mode::RAM) lock.push_back(ple_table);
        size_t bytes = 0;
        std::string names;
        for (const Tensor* t : lock) {
            lock_in_ram(*t);
            bytes += t->nbytes;
            names += (names.empty() ? "" : " and ") + t->name;
        }
        std::fprintf(stderr, "fnx: %s (%.1f GB) locked in RAM in %.1f s\n", names.c_str(), bytes / 1e9,
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        // the lab's holder runs no engine: no read pool, no cache, no probe of the disk (each worker makes its own)
        if (how != Load::HOLD) {
            ple_rows = std::make_unique<PleRows>(g, *ple_table, mode);
            std::fprintf(stderr, "fnx: %s\n", ple_rows->describe().c_str());
            if (mode == PleRows::Mode::DISK) std::fprintf(stderr, "fnx: n-gram table: its disk, %s\n", ple_rows->probe(2048).c_str());
        }
    }
    for (int64_t v : g.ia(p + "ple.layer_multipliers")) ple.mult.push_back((uint64_t) v);
    for (int64_t v : g.ia(p + "ple.head_offsets")) ple.off.push_back((uint64_t) v);
    for (int64_t v : g.ia(p + "ple.head_vocab_sizes")) ple.vocab.push_back((uint64_t) v);
    // the gather reads ple.mult[0..PLE_ORDERS) and a head h at mixed % ple.vocab[h] + ple.off[h]: the arrays must be that
    // long and every head's range must stay inside the table, or the gather reads rows (or zero-modulo) the metadata invented
    if (ple.mult.size() != PLE_ORDERS || ple.off.size() != PLE_HEADS || ple.vocab.size() != PLE_HEADS)
        throw std::runtime_error("fnx: the PLE metadata holds " + std::to_string(ple.mult.size()) + " multipliers and " +
                                 std::to_string(ple.off.size()) + " offsets and " + std::to_string(ple.vocab.size()) +
                                 " vocab sizes, want " + std::to_string(PLE_ORDERS) + " and " + std::to_string(PLE_HEADS) + " and " +
                                 std::to_string(PLE_HEADS));
    {
        // a zero multiplier drops its n-gram from the hash silently: the gather still runs, on fewer orders than the model trained with
        for (int j = 0; j < PLE_ORDERS; ++j)
            if (!ple.mult[j]) throw std::runtime_error("fnx: PLE multiplier " + std::to_string(j) + " is 0, which drops that n-gram from the hash");
        static_assert(PLE_HEADS * PLE_DIM == N, "the gather tiles PLE_HEADS rows of PLE_DIM into one embedding row");
        if (ple_table->ne[0] != PLE_DIM)
            throw std::runtime_error("fnx: " + ple_table->name + " is " + std::to_string(ple_table->ne[0]) + " a row, not the " +
                                     std::to_string(PLE_DIM) + " the gather tiles into the embedding");
        const uint64_t rows = (uint64_t) ple_table->ne[1];   // the gather reads rows of per_layer_token_embd, not the layer's key table
        for (int h = 0; h < PLE_HEADS; ++h) {
            // head h gathers rows [off, off+vocab): bound each side on its own so a negative offset (which wraps huge)
            // trips the check instead of slipping past a summed top
            if (!ple.vocab[h]) throw std::runtime_error("fnx: PLE head " + std::to_string(h) + " has an empty vocabulary");
            if (ple.off[h] >= rows || ple.vocab[h] > rows - ple.off[h])
                throw std::runtime_error("fnx: PLE head " + std::to_string(h) + " gathers from row " + std::to_string((int64_t) ple.off[h]) +
                                         " for " + std::to_string(ple.vocab[h]) + ", past " + ple_table->name + "'s " + std::to_string(rows));
        }
    }
    ple.eos = g.i(p + "ple.eos_token_id");
    ple_image = g.i(p + "ple.image_token_id");
#endif
    cards.resize(NCARD);
    // key heads per card (their q and k), value heads (numbered s * NKH + k, listed ascending: the order the card's
    // slices are laid out in) and FF slices, by card_split.hpp's rule
    int kparts[NCARD][2], ffparts[NCARD][2];
    std::vector<std::vector<int>> vparts(NCARD);
    for (int c = 0; c < NCARD; ++c) {
        kparts[c][0] = khead_first(NCARD, c);
        kparts[c][1] = khead_count(NCARD, c);
        ffparts[c][0] = ff_lo(NCARD, c);
        ffparts[c][1] = ff_hi(NCARD, c);
    }
    for (int v = 0; v < NVH; ++v) vparts[(size_t) vhead_card(NCARD, v)].push_back(v);
    for (int c = 0; c < NCARD; ++c) {
        const std::string slice = "card " + std::to_string(c) + "'s FF slice [" + std::to_string(ffparts[c][0]) + ", " +
                                  std::to_string(ffparts[c][1]) + ")";
        if (ffparts[c][0] % 32 || ffparts[c][1] % 32) throw std::runtime_error("fnx: " + slice + " is not whole 32-value blocks");
        if (ffparts[c][1] - ffparts[c][0] > FF_CARD_MAX)
            throw std::runtime_error("fnx: " + slice + " is wider than FF_CARD_MAX (" + std::to_string(FF_CARD_MAX) + ")");
        if (ffparts[c][0] != (c ? ffparts[c - 1][1] : 0)) throw std::runtime_error("fnx: " + slice + " does not start where the last ended");
    }
    if (ffparts[NCARD - 1][1] != FF) throw std::runtime_error("fnx: the cards' FF slices do not cover the FF width");
    {   // every value head on exactly one card, beside its key head (value head v belongs to key head v % NKH)
        std::vector<int> seen(NVH, 0);
        for (int c = 0; c < NCARD; ++c)
            for (const int v : vparts[(size_t) c]) {
                if (v < 0 || v >= NVH) throw std::runtime_error("fnx: card " + std::to_string(c) + " lists value head " + std::to_string(v));
                ++seen[(size_t) v];
                const int k = v % NKH;
                if (k < kparts[c][0] || k >= kparts[c][0] + kparts[c][1])
                    throw std::runtime_error("fnx: card " + std::to_string(c) + " has value head " + std::to_string(v) + " without its key head " +
                                             std::to_string(k));
            }
        for (int v = 0; v < NVH; ++v)
            if (seen[(size_t) v] != 1)
                throw std::runtime_error("fnx: value head " + std::to_string(v) + " is on " + std::to_string(seen[(size_t) v]) + " cards");
    }
    const int vstep = (vocab + NCARD - 1) / NCARD;
    for (int c = 0; c < NCARD; ++c) {
        CardW& cw = cards[c];
        cw.dev = c;
        cw.k0 = kparts[c][0];
        cw.nk = kparts[c][1];
        cw.vheads = vparts[c];
        cw.q0 = c * (NQH / NCARD);
        cw.nq = NQH / NCARD;
        cw.kv0 = cw.q0 / (NQH / NKVH);
        cw.nkv = (cw.q0 + cw.nq - 1) / (NQH / NKVH) - cw.kv0 + 1;
        cw.ff0 = ffparts[c][0];
        cw.ff1 = ffparts[c][1];
        cw.v0 = c * vstep;
        cw.v1 = std::min(vocab, (c + 1) * vstep);
    }
#if FNX_27B
    for (int c = 0; c < NCARD; ++c) cards[c].fd0 = ffd_lo(NCARD, c), cards[c].fd1 = ffd_lo(NCARD, c + 1);
    if (cards[NCARD - 1].fd1 != FFD) throw std::runtime_error("27b: the cards' dense FF slices do not cover its width");
    {   // the image-pad token, whose runs in a prompt are a picture's cells (tools/sfnx.cpp, GENI)
        const std::vector<std::string>& toks = g.tokens();
        const auto it = std::find(toks.begin(), toks.end(), "<|image_pad|>");
        if (it == toks.end()) throw std::runtime_error("27b: the tokenizer has no <|image_pad|> token");
        ple_image = it - toks.begin();
        // the ids the engine process stops and starts turns on (tools/sfnx.cpp's EOS and 248045, front/engine_limits.py's
        // EOS_IDS), which are Flash-Next's tokenizer's: a GGUF tokenized otherwise would run past its turns' ends
        for (const auto& [id, want] : {std::pair{248044, "<|endoftext|>"}, std::pair{248045, "<|im_start|>"}, std::pair{248046, "<|im_end|>"}})
            if ((size_t) id >= toks.size() || toks[(size_t) id] != want)
                throw std::runtime_error("27b: token " + std::to_string(id) + " is '" + ((size_t) id < toks.size() ? toks[(size_t) id] : "") +
                                         "', the engine stops on it as " + want);
    }
    if (how != Load::PLAIN) throw std::runtime_error("27b: plain loads only (the lab's holder does not hold the 27B)");
#endif
    if (how == Load::ATTACH) {   // a lab worker: the holder's weights are mapped already; take its tables, place the experts
        if (!image) throw std::runtime_error("fnx: attaching needs the holder's image");
        if (!store) throw std::runtime_error("fnx: attaching needs the store the holder's memory is mapped in");
        restore(*image, *store);
        place_experts();
        return;
    }
    if (how == Load::HOLD && !store) throw std::runtime_error("fnx: holding needs a store");
#if !FNX_27B
    if (how == Load::PLAIN) choose_residency();   // the holder leaves that to its workers: every expert to a host copy
    // FNX_CPU_EXPERTS=1: pre-allocate ONE single contiguous host buffer for cold experts ONLY (resident[l][e] == 0),
    // laid out as contiguous 2-half slices [card0_gu | card1_gu] and [card0_dn | card1_dn]. Registered via
    // cudaHostRegister(Portable | Mapped) right after load_card populates all cold slices in parallel.
    if (how == Load::PLAIN && cpu_experts) {
        cold_experts.assign((size_t) NL * NEXP, ColdExpertInfo{});
        const int ff_half = cards[0].ff1 - cards[0].ff0;
        if (NCARD != 2 || cards[1].ff1 - cards[1].ff0 != ff_half)
            throw std::runtime_error("fnx: FNX_CPU_EXPERTS=1 requires symmetric 2-card FF split");
        auto align64 = [](size_t x) -> size_t { return (x + 63u) & ~size_t(63); };
        struct Offs { size_t gu0 = 0, gu1 = 0, dn0 = 0, dn1 = 0; };
        std::vector<Offs> offs((size_t) NL * NEXP);
        size_t total = 0;
        for (int l = 0; l < NL; ++l) {
            const std::string b = "blk." + std::to_string(l) + ".";
            const Tensor& gt = g.t(b + "ffn_gate_exps.weight");
            const Tensor& ut = g.t(b + "ffn_up_exps.weight");
            const Tensor& dt = g.t(b + "ffn_down_exps.weight");
            if (gt.type != ut.type || gt.ne[0] != N || gt.ne[1] != FF || gt.ne[2] != NEXP ||
                ut.ne[0] != N || ut.ne[1] != FF || ut.ne[2] != NEXP ||
                dt.ne[0] != FF || dt.ne[1] != N || dt.ne[2] != NEXP)
                throw std::runtime_error("fnx: layer " + std::to_string(l) + " routed expert tensor shape mismatch");
            const TypeInfo dti = block32(dt);
            const size_t gu_rb = gt.row_bytes();
            const size_t dn_rb = (size_t) (ff_half / dti.block) * dti.bytes;
            const size_t gu_hb = (size_t) (2 * ff_half) * gu_rb;
            const size_t dn_hb = (size_t) N * dn_rb;
            for (int e = 0; e < NEXP; ++e) {
                const size_t idx = (size_t) l * NEXP + (size_t) e;
                ColdExpertInfo& ci = cold_experts[idx];
                ci.gu_type = gt.type;
                ci.dn_type = dt.type;
                ci.gu_row_bytes = gu_rb;
                ci.dn_row_bytes = dn_rb;
                ci.gu_half_bytes = gu_hb;
                ci.dn_half_bytes = dn_hb;
                ci.is_cold = !resident[(size_t) l][(size_t) e];
                if (ci.is_cold) {
                    total = align64(total);
                    offs[idx].gu0 = total;
                    total += gu_hb;
                    offs[idx].gu1 = total;
                    total += gu_hb;
                    total = align64(total);
                    offs[idx].dn0 = total;
                    total += dn_hb;
                    offs[idx].dn1 = total;
                    total += dn_hb;
                }
            }
        }
        if (total > 0) {
            const size_t alloc_bytes = (total + (2ull << 20) - 1) & ~((2ull << 20) - 1);
            void* base = mmap(nullptr, alloc_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (base == MAP_FAILED)
                throw std::runtime_error("fnx: mmap failed for " + std::to_string(alloc_bytes >> 20) + " MiB cold expert buffer: " +
                                         std::strerror(errno));
            madvise(base, alloc_bytes, MADV_HUGEPAGE);
            cold_host_via_mmap = true;
            cold_host_allocs.push_back(base);
            cold_host_alloc_sizes.push_back(alloc_bytes);
            cold_host_bytes = total;
            const uint8_t* bp = (const uint8_t*) base;
            for (size_t idx = 0; idx < cold_experts.size(); ++idx) {
                if (!cold_experts[idx].is_cold) continue;
                cold_experts[idx].gu_half[0] = bp + offs[idx].gu0;
                cold_experts[idx].gu_half[1] = bp + offs[idx].gu1;
                cold_experts[idx].dn_half[0] = bp + offs[idx].dn0;
                cold_experts[idx].dn_half[1] = bp + offs[idx].dn1;
            }
        }
    }
#endif
    const std::string clusters_file =
#if FNX_27B
        lmhead_6p2_enabled() ? resolve_vhead_clusters_path(path) : "";
#else
        "";
#endif
    const auto t0 = std::chrono::steady_clock::now();
    try {
        Stager st(g, stage_groups(g), NCARD);
        std::vector<std::thread> th;
        std::vector<std::string> errs(NCARD);
        for (int c = 0; c < NCARD; ++c)
            th.emplace_back([&, c] {
                try {
                    load_card(st, cards[c], c, resident, arena ? &score : nullptr, how == Load::HOLD ? store : nullptr,
                              how == Load::HOLD || (!resident.empty() && !cpu_experts),
                              cpu_experts ? &cold_experts : nullptr, clusters_file);
                } catch (const std::exception& e) {
                    errs[c] = e.what();
                    st.abort("card " + std::to_string(c) + " failed");   // the other cards wait on groups no one gives back
                }
            });
        for (auto& t : th) t.join();
        throw_card_errors(errs);
#if !FNX_27B
        if (cpu_experts && !cold_host_allocs.empty()) {
            const auto tp0 = std::chrono::steady_clock::now();
            void* base = cold_host_allocs[0];
            const size_t alloc_bytes = cold_host_alloc_sizes[0];
            ck(cudaSetDevice(cards[0].dev), "cudaSetDevice");
            const cudaError_t re = cudaHostRegister(base, alloc_bytes, cudaHostRegisterPortable | cudaHostRegisterMapped);
            if (re != cudaSuccess) {
                cudaGetLastError();
                std::fprintf(stderr, "fnx: cudaHostRegister(%.2f GiB) failed (%s); falling back to cudaHostAlloc(Portable|Mapped)\n",
                             alloc_bytes / (double) (1ull << 30), cudaGetErrorString(re));
                void* pinned = nullptr;
                ck(cudaHostAlloc(&pinned, alloc_bytes, cudaHostAllocPortable | cudaHostAllocMapped), "cold experts cudaHostAlloc");
                std::memcpy(pinned, base, cold_host_bytes);
                const intptr_t delta = (intptr_t) ((uint8_t*) pinned - (uint8_t*) base);
                for (ColdExpertInfo& ci : cold_experts) {
                    if (!ci.is_cold) continue;
                    for (int h = 0; h < 2; ++h) {
                        ci.gu_half[h] += delta;
                        ci.dn_half[h] += delta;
                    }
                }
                munmap(base, alloc_bytes);
                cold_host_allocs[0] = pinned;
                cold_host_via_mmap = false;
                base = pinned;
            }
            for (int c = 0; c < NCARD; ++c) {
                ck(cudaSetDevice(cards[c].dev), "cudaSetDevice");
                void* dp = nullptr;
                ck(cudaHostGetDevicePointer(&dp, base, 0), "cold experts cudaHostGetDevicePointer");
                if (dp != base)
                    throw std::runtime_error("fnx: cold expert host buffer is not UVA-identical on card " + std::to_string(c));
            }
            cards[0].host_bytes = cold_host_bytes;
            cards[1].host_bytes = 0;
            const double pin_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - tp0).count();
            std::fprintf(stderr, "fnx: cold expert buffer %.2f GiB pinned & UVA-mapped (%s) in %.2f s\n",
                         cold_host_bytes / (double) (1ull << 30), cold_host_via_mmap ? "mmap+THP+cudaHostRegister" : "cudaHostAlloc",
                         pin_s);
        }
#endif
        // the disk rate only when the disk gave nearly all of it (tools/lab.sh estimates its next cold loads from it)
        const double disk = st.disk_bytes() / 1e9, cache = st.cache_bytes() / 1e9;
        char rate[64] = "";
        if (cache < 0.05 * (disk + cache)) std::snprintf(rate, sizeof rate, " at %.2f GB/s", disk / st.read_seconds());
        std::fprintf(stderr, "fnx: weights read in %.1f s of reading (%.1f GB from the disk%s, %.1f GB from the page cache), placed in %.1f s\n",
                     st.read_seconds(), disk, rate, cache, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    } catch (...) {
        release_cold_host_allocs(*this);
        throw;
    }
}

void Model::choose_residency() {
    resident.clear();
    score.clear();
    // prod's settings are the defaults: the arena, the coldest 3 GiB of experts in RAM, ranked by
    // data/expert-routes.bin (expert use tallied over an agentic code-review panel).
    // On FNX_ARCH < 70 && NCARD == 2 (Dual-P6000 Z420): FNX_CPU_EXPERTS defaults to 1 (static hot VRAM residency +
    // single un-duplicated pinned cold host buffer, swap=0) and FNX_GPU1_RESERVE_MIB defaults to 2048 MiB.
    const long long gpu1_res_def = (FNX_ARCH < 70 && NCARD == 2) ? 2048 : 0;
    const long long gpu1_res_mib = env_int("FNX_GPU1_RESERVE_MIB", gpu1_res_def);
    if (gpu1_res_mib < 0)
        throw std::runtime_error("fnx: FNX_GPU1_RESERVE_MIB: want MiB >= 0, got " + std::to_string(gpu1_res_mib));
    gpu1_reserve_bytes = (size_t) gpu1_res_mib << 20;

    const char* ram_s = std::getenv("FNX_EXPERT_RAM_GIB");
    const char* arena_s = std::getenv("FNX_ARENA");
    const char* cpu_s = std::getenv("FNX_CPU_EXPERTS");
    if (arena_s && std::strcmp(arena_s, "0") && std::strcmp(arena_s, "1"))
        throw std::runtime_error(std::string("fnx: FNX_ARENA: 0 or 1, not '") + arena_s + "'");
    if (cpu_s && std::strcmp(cpu_s, "0") && std::strcmp(cpu_s, "1"))
        throw std::runtime_error(std::string("fnx: FNX_CPU_EXPERTS: 0 or 1, not '") + cpu_s + "'");
    const bool cache = !(ram_s && !std::strcmp(ram_s, "off"));
    double ram_gib = 3.0;
    if (ram_s && cache) {
        char* end = nullptr;
        ram_gib = std::strtod(ram_s, &end);
        if (end == ram_s || *end || ram_gib < 0)
            throw std::runtime_error(std::string("fnx: FNX_EXPERT_RAM_GIB: GiB (0 or more) or off, not '") + ram_s + "'");
    }
    const bool default_cpu = (!FNX_27B && FNX_ARCH < 70 && NCARD == 2 && cache);
    cpu_experts = cpu_s ? !std::strcmp(cpu_s, "1") : default_cpu;
    if (cpu_experts && NCARD != 2) throw std::runtime_error("fnx: FNX_CPU_EXPERTS=1 requires FNX_NCARD=2");
    if (cpu_experts && !cache) {
        if (cpu_s) throw std::runtime_error("fnx: FNX_CPU_EXPERTS=1 cannot be combined with FNX_EXPERT_RAM_GIB=off");
        cpu_experts = false;
    }
    if (cpu_experts) {
        if (arena_s && !std::strcmp(arena_s, "1"))
            throw std::runtime_error("fnx: FNX_CPU_EXPERTS=1 uses static hot VRAM residency and cannot be combined with FNX_ARENA=1");
        arena = false;
    } else {
        arena = !arena_s || !std::strcmp(arena_s, "1");
    }
    if (arena && !cache) throw std::runtime_error("fnx: FNX_EXPERT_RAM_GIB=off needs FNX_ARENA=0 (the arena lends the cache's slots)");

    if (cache) {
        std::vector<double> bytes(NL);
        double total_expert_bytes = 0.0;
        for (int l = 0; l < NL; ++l) {
            const std::string b = "blk." + std::to_string(l) + ".";
            const double lb = (double) (g.t(b + "ffn_gate_exps.weight").nbytes + g.t(b + "ffn_up_exps.weight").nbytes +
                                        g.t(b + "ffn_down_exps.weight").nbytes);
            bytes[(size_t) l] = lb / NEXP;
            total_expert_bytes += lb;
        }
        const char* prof_s = std::getenv("FNX_EXPERT_PROFILE");
        const std::string prof_d = std::string(FNX_DIR) + "/data/expert-routes.bin";
        const char* prof = prof_s ? prof_s : prof_d.c_str();
        load_expert_profile(prof, bytes, score);

        double want = ram_gib * (double) (1ull << 30);
        if ((gpu1_reserve_bytes > 0 || cpu_experts) && cards.size() == (size_t) NCARD) {
            // Factor live free VRAM across all cards + GPU 1 display reserve + non-routed weights + KV/prefill/MTP/graphs
            const size_t non_routed = estimate_non_routed_per_card(g);
            const long long stage_mib = env_int("FNX_PF_STAGE_MIB", cpu_experts ? 128 : 384);
            const long long eng_base_mib = env_int("FNX_ENGINE_RESERVE_MIB", 2800);
            const long long base_res_mib = env_int("FNX_ARENA_RESERVE_MIB", 512);
            long long mtp_mib = 0;
            if (std::getenv("FNX_MTP_RESERVE_MIB")) {
                mtp_mib = env_int("FNX_MTP_RESERVE_MIB", 0);
            } else if (const char* mg = std::getenv("FNX_MTP_GGUF"); mg && *mg) {
                mtp_mib = 1536;
            } else if (const char* md = std::getenv("FNX_MTP_DIR"); md && *md) {
                mtp_mib = 900;
            } else if (FNX_ARCH < 70 && NCARD == 2) {
                mtp_mib = 900;
            }
            if (stage_mib < 0 || eng_base_mib < 0 || base_res_mib < 0 || mtp_mib < 0)
                throw std::runtime_error("fnx: negative VRAM reserve setting");
            const size_t overhead = (size_t) (eng_base_mib + 2 * stage_mib + mtp_mib) << 20;
            size_t min_avail = SIZE_MAX;
            for (int c = 0; c < NCARD; ++c) {
                ck(cudaSetDevice(cards[c].dev), "cudaSetDevice");
                ck(cudaFree(nullptr), "cuda init");
                size_t fr = 0, tot = 0;
                ck(cudaMemGetInfo(&fr, &tot), "cudaMemGetInfo");
                size_t card_res = (size_t) base_res_mib << 20;
                if (c == 1 && gpu1_reserve_bytes > card_res) card_res = gpu1_reserve_bytes;
                const size_t non_exp = (cards[c].bytes == 0 ? non_routed : 0) + overhead + card_res;
                const size_t avail = fr > non_exp ? fr - non_exp : 0;
                min_avail = std::min(min_avail, avail);
            }
            const double max_vram_experts = (double) NCARD * (double) min_avail;
            const double auto_min_ram = total_expert_bytes > max_vram_experts ? total_expert_bytes - max_vram_experts : 0.0;
            if (auto_min_ram > want) {
                std::fprintf(stderr,
                             "fnx: VRAM budget (GPU1 reserve %zu MiB, overhead %zu MiB/card, non-routed %zu MiB/card, "
                             "avail %.2f GiB/card for hot experts): keeping >= %.2f GiB of routed experts in host RAM\n",
                             gpu1_reserve_bytes >> 20, overhead >> 20, non_routed >> 20,
                             min_avail / (double) (1ull << 30), auto_min_ram / (double) (1ull << 30));
                want = auto_min_ram;
            }
        }

        std::vector<int> order((size_t) NL * NEXP);
        for (int i = 0; i < NL * NEXP; ++i) order[(size_t) i] = i;
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return score[(size_t) a] < score[(size_t) b]; });
        resident.assign(NL, std::vector<char>(NEXP, 1));
        double moved = 0;
        int n = 0, extra = 0;
        for (int i : order) {
            if (moved >= want && extra++ >= ram_extra) break;
            resident[(size_t) (i / NEXP)][(size_t) (i % NEXP)] = 0;
            moved += bytes[(size_t) (i / NEXP)];
            ++n;
        }
        std::fprintf(stderr, "fnx: %d of %d routed experts (%.2f GiB) in RAM (%s), %d (%.2f GiB) in VRAM, ranked by %s%s\n",
                     n, NL * NEXP, moved / (double) (1ull << 30), cpu_experts ? "FNX_CPU_EXPERTS=1" : "ExpertCache",
                     NL * NEXP - n, (total_expert_bytes - moved) / (double) (1ull << 30), prof,
                     ram_extra ? (" (" + std::to_string(ram_extra) + " of them because VRAM was short)").c_str() : "");
    }
}

namespace {
template <typename F>
void each_card(std::vector<CardW>& cards, F f) {
    std::vector<std::thread> th;
    std::vector<std::string> errs(NCARD);
    for (int c = 0; c < NCARD; ++c)
        th.emplace_back([&, c] {
            try {
                ck(cudaSetDevice(cards[c].dev), "cudaSetDevice");
                f(cards[c]);
            } catch (const std::exception& e) { errs[c] = e.what(); }
        });
    for (auto& t : th) t.join();
    throw_card_errors(errs);
}
}  // namespace

void Model::place_experts() {
    choose_residency();
    if (cpu_experts) throw std::runtime_error("fnx: place_experts() is for Load::ATTACH (ExpertCache), not FNX_CPU_EXPERTS=1");
    each_card(cards, [&](CardW& c) {
        if (c.ec_host_gu.size() != (size_t) NL * NEXP) throw std::runtime_error("fnx: placing experts needs their host copies");
        const size_t before = c.bytes;
        Up up{c, nullptr, &c.expert_mem};
        if (resident.empty()) {   // every expert in VRAM: each layer's host copy is one block
            for (int l = 0; l < NL; ++l) {
                LayerW& L = c.L[(size_t) l];
                L.gu_e.p = up.put<uint8_t>(c.ec_host_gu[(size_t) l * NEXP], (size_t) NEXP * L.gu_e_stride);
                L.down_e.p = up.put<uint8_t>(c.ec_host_dn[(size_t) l * NEXP], (size_t) NEXP * L.down_e_stride);
            }
        } else {
            const std::vector<const char*> hgu = c.ec_host_gu, hdn = c.ec_host_dn;
            up.expert_cache(resident, hgu, hdn, arena ? &score : nullptr);
        }
        c.expert_bytes = c.bytes - before;
    });
}

void Model::free_experts() {
    each_card(cards, [&](CardW& c) {
        for (void* p : c.expert_mem) ck(cudaFree(p), "free experts");
        c.expert_mem.clear();
        c.bytes -= c.expert_bytes;
        c.expert_bytes = 0;
        c.ec = ExpertCache{};
        c.resident_mask = nullptr;
        c.ec_owner0.clear();
        c.ec_slot0.clear();
        for (LayerW& L : c.L) {
            L.gu_e.p = L.down_e.p = nullptr;
            L.gu_e.ep = L.down_e.ep = nullptr;
        }
    });
}

// ---------------------------------------------------------------- the holder's tables for a worker
// Per card what load_card stored (the geometry the constructor sets again): its pointers, the layers, the experts'
// host copies. card_fields lists them once for both directions; a field added to CardW without it fails the size check.
namespace {
static_assert(std::is_trivially_copyable<LayerW>::value && std::is_trivially_copyable<DW>::value, "layers copied as bytes");
struct Img {
    std::string s;
    size_t at = 0;
    template <typename T> void put(const T& v) { s.append((const char*) &v, sizeof v); }
    template <typename T> void get(T& v) {
        if (at + sizeof v > s.size()) throw std::runtime_error("fnx: the holder's image is short");
        std::memcpy((void*) &v, s.data() + at, sizeof v);
        at += sizeof v;
    }
    template <typename T> void put_vec(const std::vector<T>& v) {
        put((uint64_t) v.size());
        s.append((const char*) v.data(), v.size() * sizeof(T));
    }
    template <typename T> void get_vec(std::vector<T>& v) {
        uint64_t n = 0;
        get(n);
        if (at + n * sizeof(T) > s.size()) throw std::runtime_error("fnx: the holder's image is short");
        v.resize(n);
        std::memcpy((void*) v.data(), s.data() + at, n * sizeof(T));
        at += n * sizeof(T);
    }
};
template <typename F>
void card_fields(CardW& c, F f) {
    f(c.vmap), f(c.head_norm), f(c.head_down), f(c.head_up), f(c.out), f(c.out_6p2), f(c.out_cent_f16), f(c.out_cluster_off),
        f(c.out_n_clusters), f(c.ple_key), f(c.ple_value), f(c.ple_norm_key), f(c.ple_norm_query), f(c.ple_norm_conv), f(c.ple_conv),
        f(c.bytes), f(c.host_bytes), f(c.resident_mask);
}
#if !FNX_27B   // the 27B has no holder (plain loads only)
static_assert(sizeof(CardW) == 984, "CardW changed: put what load_card stores in card_fields, then update this size");
#endif
// every address load_card stores, in a card and in a layer: a worker moves them to where it mapped the holder's memory
template <typename F>
void card_ptrs(CardW& c, F f) {
    auto q8 = [&](Q8S& s) { f(s.q), f(s.d); };
    auto dw = [&](DW& d) { f(d.p), q8(d.s), f(d.ep); };
    auto vh = [&](VHead6p2& v) { f(v.hi), f(v.lo), f(v.d), f(v.row_meta), f(v.perm_to_vocab), f(v.vocab_to_local); };
    f(c.vmap), f(c.head_norm), q8(c.head_down), q8(c.head_up), dw(c.out), vh(c.out_6p2), f(c.out_cent_f16), f(c.out_cluster_off),
        dw(c.ple_key), dw(c.ple_value), f(c.ple_norm_key), f(c.ple_norm_query), f(c.ple_norm_conv), f(c.ple_conv), f(c.resident_mask);
    for (LayerW& w : c.L) {
        for (int h = 0; h < 2; ++h) f(w.hc_norm[h]), q8(w.hc_down[h]), q8(w.hc_up[h]), q8(w.hc_inj[h]);
        f(w.router), dw(w.qkv), dw(w.z), dw(w.ssm_out), dw(w.qkvz), f(w.ab), f(w.conv), f(w.alpha), f(w.beta), f(w.ssm_a),
            f(w.dt_bias), f(w.ssm_norm);
        dw(w.wq), dw(w.wk), dw(w.wv), dw(w.wo), f(w.q_norm), f(w.k_norm), f(w.idx_wq), f(w.idx_wk), dw(w.idx_qk), f(w.idx_qn),
            f(w.idx_kn);
        dw(w.gu_e), dw(w.down_e), dw(w.sh_gu), dw(w.sh_down);
    }
    for (const char*& p : c.ec_host_gu) f(p);
    for (const char*& p : c.ec_host_dn) f(p);
}
#if !FNX_27B
static_assert(sizeof(LayerW) == 968, "LayerW changed: put its new addresses in card_ptrs, then update this size");
#endif
}  // namespace

std::string Model::image() const {
    Img m;
    m.put((uint32_t) 0x31474d49);   // "IMG1"
    for (const CardW& cc : cards) {
        CardW& c = const_cast<CardW&>(cc);
        card_fields(c, [&](auto& v) { m.put(v); });
        m.put_vec(c.L);
        m.put_vec(c.ec_host_gu);
        m.put_vec(c.ec_host_dn);
    }
    return m.s;
}

void Model::restore(const std::string& image, const Store& store) {
    Img m{image};
    uint32_t magic = 0;
    m.get(magic);
    if (magic != 0x31474d49) throw std::runtime_error("fnx: not a holder's image");
    for (CardW& c : cards) {
        card_fields(c, [&](auto& v) { m.get(v); });
        m.get_vec(c.L);
        m.get_vec(c.ec_host_gu);
        m.get_vec(c.ec_host_dn);
        if (c.L.size() != (size_t) NL) throw std::runtime_error("fnx: the holder's image has another layer count");
        card_ptrs(c, [&](auto& p) { store.move_ptr(p); });
    }
    if (m.at != image.size()) throw std::runtime_error("fnx: the holder's image is longer than its cards");
}

}  // namespace fnx
