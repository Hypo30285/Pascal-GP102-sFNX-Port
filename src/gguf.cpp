#include "gguf.hpp"

#include <dirent.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <regex>
#include <set>
#include <stdexcept>

namespace fnx {

TypeInfo type_info(uint32_t t) {
    switch (t) {
        case T_F32: return {1, 4};
        case T_F16: return {1, 2};
        case T_BF16: return {1, 2};
        case T_Q4_0: return {32, 18};
        case T_Q4_1: return {32, 20};
        case T_Q5_0: return {32, 22};
        case T_Q5_1: return {32, 24};
        case T_Q8_0: return {32, 34};
        case T_Q8_1: return {32, 36};
        case T_Q2_K: return {256, 84};
        case T_Q3_K: return {256, 110};
        case T_Q4_K: return {256, 144};
        case T_Q5_K: return {256, 176};
        case T_Q6_K: return {256, 210};
        case T_IQ4_NL: return {32, 18};
    }
    throw std::runtime_error("gguf: unknown ggml type " + std::to_string(t));
}

size_t Tensor::row_bytes() const {
    const TypeInfo ti = type_info(type);
    return (size_t) (ne[0] / ti.block) * ti.bytes;
}

namespace {
struct Cur {
    const uint8_t* p;
    const uint8_t* end;
    template <typename T> T get() {
        if (p + sizeof(T) > end) throw std::runtime_error("gguf: truncated header");
        T v;
        std::memcpy(&v, p, sizeof(T));
        p += sizeof(T);
        return v;
    }
    std::string str() {
        const uint64_t n = get<uint64_t>();
        if (p + n > end) throw std::runtime_error("gguf: truncated string");
        std::string s((const char*) p, n);
        p += n;
        return s;
    }
};

int scalar_size(uint32_t t) {
    switch (t) {
        case 0: case 1: case 7: return 1;
        case 2: case 3: return 2;
        case 4: case 5: case 6: return 4;
        case 10: case 11: case 12: return 8;
    }
    return -1;
}

int64_t read_int(Cur& c, uint32_t t) {
    switch (t) {
        case 0: return c.get<uint8_t>();
        case 1: return c.get<int8_t>();
        case 2: return c.get<uint16_t>();
        case 3: return c.get<int16_t>();
        case 4: return c.get<uint32_t>();
        case 5: return c.get<int32_t>();
        case 7: return c.get<uint8_t>();
        case 10: return (int64_t) c.get<uint64_t>();
        case 11: return c.get<int64_t>();
    }
    throw std::runtime_error("gguf: not an integer type");
}
}  // namespace

namespace {
bool is_q4q6q8_full_layer(int l) {
    return l == 0 || l == 1 || l == 2 || l == 4 || l == 30 || l == 46 || l == 47;
}

bool same_shape(const Tensor& a, const Tensor& b) {
    if (a.n_dims != b.n_dims) return false;
    for (int d = 0; d < 4; ++d)
        if (a.ne[d] != b.ne[d]) return false;
    return true;
}

std::string shape_str(const Tensor& t) {
    std::string s = "[";
    for (int d = 0; d < t.n_dims; ++d) s += (d ? ", " : "") + std::to_string(t.ne[d]);
    return s + "]";
}
}  // namespace

Gguf::Gguf(const std::string& first, bool apply_env) {
    GgufOpts opts;
    if (apply_env) {
        if (const char* e = std::getenv("FNX_ENGRAM_GGUF"); e && *e) opts.engram_gguf = e;
        if (const char* h = std::getenv("FNX_HYBRID_Q6_GGUF"); h && *h) opts.hybrid_q6_gguf = h;
        if (const char* hm = std::getenv("FNX_HYBRID_MAP"); hm && *hm) opts.hybrid_map = hm;
    }
    try {
        init(first, opts);
    } catch (...) {
        close_maps();
        throw;
    }
}

Gguf::Gguf(const std::string& first, const GgufOpts& opts) {
    try {
        init(first, opts);
    } catch (...) {
        close_maps();
        throw;
    }
}

void Gguf::close_maps() {
    for (Map& mp : maps_) {
        if (mp.base && mp.base != MAP_FAILED) munmap(mp.base, mp.size);
        if (mp.fd >= 0) close(mp.fd);
    }
    maps_.clear();
}

Gguf::~Gguf() { close_maps(); }

void Gguf::init(const std::string& first, const GgufOpts& opts) {
    const bool has_engram = !opts.engram_gguf.empty();
    std::smatch m;
    static const std::regex split(R"((.*-)(\d{5})(-of-)(\d{5})(\.gguf)$)");
    if (std::regex_match(first, m, split)) {
        const int n = std::stoi(m[4]);
        for (int i = 1; i <= n; ++i) {
            char num[16];
            std::snprintf(num, sizeof num, "%05d", i);
            const std::string sp = m[1].str() + num + m[3].str() + m[4].str() + m[5].str();
            if (has_engram && n > 1 && i == n && access(sp.c_str(), F_OK) != 0 && errno == ENOENT) {
                std::fprintf(stderr, "gguf: skipping missing final split shard %s (FNX_ENGRAM_GGUF set)\n", sp.c_str());
                continue;
            }
            load_shard(sp, i == 1);
        }
    } else {
        load_shard(first, true);
    }

    if (has_engram) {
        std::map<std::string, Tensor> ext;
        load_shard(opts.engram_gguf, false, &ext);
        const auto it = ext.find("per_layer_token_embd.weight");
        if (it == ext.end())
            throw std::runtime_error("gguf: FNX_ENGRAM_GGUF " + opts.engram_gguf + " does not contain per_layer_token_embd.weight");
        const Tensor& et = it->second;
        if (et.n_dims != 2 || et.ne[0] <= 0 || et.ne[1] <= 0)
            throw std::runtime_error("gguf: FNX_ENGRAM_GGUF per_layer_token_embd.weight has invalid shape " + shape_str(et));
        if (const Tensor* prev = find("per_layer_token_embd.weight")) {
            if (!same_shape(*prev, et))
                throw std::runtime_error("gguf: FNX_ENGRAM_GGUF per_layer_token_embd.weight shape " + shape_str(et) +
                                         " does not match base " + shape_str(*prev));
        }
        tensors_["per_layer_token_embd.weight"] = et;
        engram_override_ = true;
        engram_path_ = opts.engram_gguf;
        std::fprintf(stderr, "gguf: FNX_ENGRAM_GGUF loaded per_layer_token_embd.weight (type=%u, ne=%lldx%lld, %.2f GiB) from %s\n",
                     et.type, (long long) et.ne[0], (long long) et.ne[1], et.nbytes / (double) (1ull << 30), opts.engram_gguf.c_str());
    }

    if (!opts.hybrid_q6_gguf.empty()) {
        const std::string hmap = opts.hybrid_map.empty() ? "q4q6q8" : opts.hybrid_map;
        if (hmap != "q4q6q8")
            throw std::runtime_error("gguf: unsupported FNX_HYBRID_MAP '" + hmap + "' (expected q4q6q8)");
        std::map<std::string, Tensor> ext;
        load_shard_set(opts.hybrid_q6_gguf, /*allow_missing=*/true, ext);
        // Discover all routed-expert layers present in the base model
        std::set<int> layers;
        for (const auto& kv : tensors_) {
            int l = -1, nch = 0;
            if (std::sscanf(kv.first.c_str(), "blk.%d.ffn_down_exps.weight%n", &l, &nch) == 1 &&
                nch == (int) kv.first.size() && l >= 0)
                layers.insert(l);
        }
        if (layers.empty())
            throw std::runtime_error("gguf: FNX_HYBRID_Q6_GGUF set, but base model has no blk.<l>.ffn_down_exps.weight tensors");
        auto splice_one = [&](const std::string& name) {
            const auto bi = tensors_.find(name);
            if (bi == tensors_.end())
                throw std::runtime_error("gguf: hybrid splice base model missing tensor " + name);
            const auto si = ext.find(name);
            if (si == ext.end())
                throw std::runtime_error("gguf: FNX_HYBRID_Q6_GGUF " + opts.hybrid_q6_gguf + " missing tensor " + name);
            if (!same_shape(bi->second, si->second))
                throw std::runtime_error("gguf: hybrid splice tensor " + name + " shape " + shape_str(si->second) +
                                         " does not match base " + shape_str(bi->second));
            bi->second = si->second;
        };
        int n_full = 0, n_down_only = 0;
        uint64_t total_expert_bytes = 0;
        int64_t total_experts = 0;
        for (int l : layers) {
            const std::string b = "blk." + std::to_string(l) + ".";
            if (is_q4q6q8_full_layer(l)) {
                splice_one(b + "ffn_gate_exps.weight");
                splice_one(b + "ffn_up_exps.weight");
                splice_one(b + "ffn_down_exps.weight");
                ++n_full;
            } else {
                splice_one(b + "ffn_down_exps.weight");
                ++n_down_only;
            }
            const Tensor& gt = t(b + "ffn_gate_exps.weight");
            const Tensor& ut = t(b + "ffn_up_exps.weight");
            const Tensor& dt = t(b + "ffn_down_exps.weight");
            total_expert_bytes += (uint64_t) gt.nbytes + ut.nbytes + dt.nbytes;
            total_experts += dt.ne[2];
        }
        hybrid_splice_ = true;
        hybrid_map_ = hmap;
        hybrid_q6_path_ = opts.hybrid_q6_gguf;
        std::fprintf(stderr,
                     "gguf: FNX_HYBRID_Q6_GGUF (%s): spliced %zu layers (%d full gate/up/down, %d down-only), "
                     "%lld experts = %llu B (%.4f GiB) from %s\n",
                     hmap.c_str(), layers.size(), n_full, n_down_only, (long long) total_experts,
                     (unsigned long long) total_expert_bytes, total_expert_bytes / (double) (1ull << 30),
                     opts.hybrid_q6_gguf.c_str());
    }
}

void Gguf::load_shard_set(const std::string& path, bool allow_missing, std::map<std::string, Tensor>& dst) {
    struct stat st{};
    if (stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) {
        DIR* dir = opendir(path.c_str());
        if (!dir) throw std::runtime_error("gguf: cannot open directory " + path);
        std::vector<std::string> files;
        for (dirent* de = readdir(dir); de; de = readdir(dir)) {
            const std::string name = de->d_name;
            if (name.size() <= 5 || name.compare(name.size() - 5, 5, ".gguf") != 0) continue;
            if (name.compare(0, 6, "mmproj") == 0 || name.compare(0, 3, "mtp") == 0) continue;
            files.push_back(path + (path.back() == '/' ? "" : "/") + name);
        }
        closedir(dir);
        std::sort(files.begin(), files.end());
        if (files.empty()) throw std::runtime_error("gguf: no .gguf shards found in directory " + path);
        for (const std::string& f : files) load_shard(f, false, &dst);
        return;
    }
    std::smatch m;
    static const std::regex split(R"((.*-)(\d{5})(-of-)(\d{5})(\.gguf)$)");
    if (std::regex_match(path, m, split)) {
        const int n = std::stoi(m[4]);
        int loaded = 0;
        for (int i = 1; i <= n; ++i) {
            char num[16];
            std::snprintf(num, sizeof num, "%05d", i);
            const std::string sp = m[1].str() + num + m[3].str() + m[4].str() + m[5].str();
            if (allow_missing && access(sp.c_str(), F_OK) != 0 && errno == ENOENT) continue;
            load_shard(sp, false, &dst);
            ++loaded;
        }
        if (loaded == 0) throw std::runtime_error("gguf: cannot open any shard of " + path);
    } else {
        load_shard(path, false, &dst);
    }
}

void Gguf::load_shard(const std::string& path, bool meta, std::map<std::string, Tensor>* dst) {
    Map mp;
    mp.path = path;
    mp.fd = open(path.c_str(), O_RDONLY);
    if (mp.fd < 0) throw std::runtime_error("gguf: cannot open " + path);
    struct stat st;
    if (fstat(mp.fd, &st) != 0) {
        close(mp.fd);
        throw std::runtime_error("gguf: cannot stat " + path);
    }
    mp.size = (size_t) st.st_size;
    mp.base = mmap(nullptr, mp.size, PROT_READ, MAP_SHARED, mp.fd, 0);
    if (mp.base == MAP_FAILED) {
        close(mp.fd);
        throw std::runtime_error("gguf: mmap failed for " + path);
    }
    maps_.push_back(mp);
    Cur c{(const uint8_t*) mp.base, (const uint8_t*) mp.base + mp.size};
    if (c.get<uint32_t>() != 0x46554747u) throw std::runtime_error("gguf: bad magic in " + path);
    c.get<uint32_t>();
    const uint64_t n_t = c.get<uint64_t>(), n_kv = c.get<uint64_t>();
    uint64_t align = 32;
    for (uint64_t i = 0; i < n_kv; ++i) {
        const std::string key = c.str();
        if (meta) keys_.insert(key);
        const uint32_t vt = c.get<uint32_t>();
        if (vt == 8) {
            std::string v = c.str();
            if (meta) strs_[key] = std::move(v);
        } else if (vt == 9) {
            const uint32_t et = c.get<uint32_t>();
            const uint64_t n = c.get<uint64_t>();
            if (et == 8) {
#if FNX_27B   // the tokenizer's tokens: the 27B finds its image-pad token there (Flash-Next's GGUF names it)
                if (meta && key == "tokenizer.ggml.tokens") {
                    std::vector<std::string> v(n);
                    for (uint64_t k = 0; k < n; ++k) v[k] = c.str();
                    str_arrays_[key] = std::move(v);
                } else
#endif
                for (uint64_t k = 0; k < n; ++k) c.str();
            } else {
                const int sz = scalar_size(et);
                if (sz < 0) throw std::runtime_error("gguf: nested array in " + key);
                if (meta && et != 6 && et != 12) {
                    std::vector<int64_t> v(n);
                    for (uint64_t k = 0; k < n; ++k) v[k] = read_int(c, et);
                    int_arrays_[key] = std::move(v);
                } else {
                    c.p += n * (uint64_t) sz;
                }
            }
        } else if (vt == 6) {
            const float v = c.get<float>();
            if (meta) floats_[key] = v;
        } else if (vt == 12) {
            const double v = c.get<double>();
            if (meta) floats_[key] = v;
        } else {
            const int64_t v = read_int(c, vt);
            if (meta) ints_[key] = v;
            if (key == "general.alignment") {
                if (v < 1) throw std::runtime_error("gguf: general.alignment " + std::to_string(v) + " in " + path);
                align = (uint64_t) v;
            }
        }
    }
    std::vector<Tensor> ts;
    std::vector<uint64_t> offs;
    for (uint64_t i = 0; i < n_t; ++i) {
        Tensor t;
        t.name = c.str();
        const uint32_t n_dims = c.get<uint32_t>();
        if (n_dims < 1 || n_dims > 4)
            throw std::runtime_error("gguf: tensor " + t.name + " has " + std::to_string(n_dims) + " dimensions in " +
                                     path + " (1 to 4 allowed)");
        t.n_dims = (int) n_dims;
        for (int d = 0; d < t.n_dims; ++d) {
            const uint64_t ne = c.get<uint64_t>();
            if (ne < 1 || ne > (uint64_t) INT64_MAX)
                throw std::runtime_error("gguf: tensor " + t.name + " has size " + std::to_string(ne) +
                                         " in dimension " + std::to_string(d) + " in " + path);
            t.ne[d] = (int64_t) ne;
        }
        t.type = c.get<uint32_t>();
        offs.push_back(c.get<uint64_t>());
        ts.push_back(t);
    }
    const uint64_t data0 = ((uint64_t) (c.p - (const uint8_t*) mp.base) + align - 1) / align * align;
    std::map<std::string, Tensor>& target = dst ? *dst : tensors_;
    for (size_t i = 0; i < ts.size(); ++i) {
        Tensor& t = ts[i];
        const TypeInfo ti = type_info(t.type);
        // element count, byte count and end offset in checked arithmetic: a wrapped product would pass the end check
        uint64_t n_el = 1, nbytes = 0, end = 0;
        for (int d = 0; d < 4; ++d)
            if (__builtin_mul_overflow(n_el, (uint64_t) t.ne[d], &n_el))
                throw std::runtime_error("gguf: tensor " + t.name + "'s element count overflows in " + path);
        if (__builtin_mul_overflow(n_el / ti.block, (uint64_t) ti.bytes, &nbytes) ||
            __builtin_add_overflow(data0, offs[i], &end) || __builtin_add_overflow(end, nbytes, &end) ||
            end > mp.size)
            throw std::runtime_error("gguf: tensor " + t.name + " runs past the end of " + path);
        t.nbytes = (size_t) nbytes;
        t.data = (const uint8_t*) mp.base + data0 + offs[i];
        t.shard = (int) maps_.size() - 1;
        t.offset = data0 + offs[i];
        target[t.name] = t;
    }
}

const Tensor* Gguf::find(const std::string& name) const {
    auto it = tensors_.find(name);
    return it == tensors_.end() ? nullptr : &it->second;
}
const Tensor& Gguf::t(const std::string& name) const {
    if (const Tensor* p = find(name)) return *p;
    throw std::runtime_error("gguf: no tensor " + name);
}
bool Gguf::has(const std::string& key) const { return ints_.count(key) || floats_.count(key) || int_arrays_.count(key); }
int64_t Gguf::i(const std::string& key) const {
    auto it = ints_.find(key);
    if (it == ints_.end()) throw std::runtime_error("gguf: no integer key " + key);
    return it->second;
}
double Gguf::f(const std::string& key) const {
    auto it = floats_.find(key);
    if (it == floats_.end()) throw std::runtime_error("gguf: no float key " + key);
    return it->second;
}
const std::string& Gguf::s(const std::string& key) const {
    auto it = strs_.find(key);
    if (it == strs_.end()) throw std::runtime_error("gguf: no string key " + key);
    return it->second;
}
#if FNX_27B
const std::vector<std::string>& Gguf::tokens() const {
    auto it = str_arrays_.find("tokenizer.ggml.tokens");
    if (it == str_arrays_.end()) throw std::runtime_error("gguf: no tokenizer.ggml.tokens");
    return it->second;
}
#endif

std::vector<int64_t> Gguf::ia(const std::string& key) const {
    auto it = int_arrays_.find(key);
    if (it == int_arrays_.end()) throw std::runtime_error("gguf: no integer array " + key);
    return it->second;
}

// ---------------------------------------------------------------- Stager
namespace {
constexpr uint64_t PAGE = 4096;
constexpr uint64_t PIECE = 16ull << 20;   // 4 threads x 16 MiB: 3.4 GB/s from the NVMe; more of either did not read faster
constexpr int READERS = 4;
constexpr uint64_t MERGE_GAP = 1ull << 20;   // tensors closer than this in a file are read as one range
}  // namespace

Stager::Stager(const Gguf& g, std::vector<Group> groups, int consumers, int ahead) : g_(g), consumers_(consumers), ahead_(ahead) {
    if (consumers < 1 || ahead < 1) throw std::runtime_error("stager: needs a consumer and a group ahead");
    bool no_direct = false;
    for (int s = 0; s < g.shards(); ++s) {
        const int pf = ::open(g.shard_path(s).c_str(), O_RDONLY);
        if (pf < 0) throw std::runtime_error("stager: cannot open " + g.shard_path(s) + ": " + std::strerror(errno));
        plain_fd_.push_back(pf);
        // a file system without O_DIRECT (tmpfs) is read through the page cache: the same bytes, only slower;
        // Z420 sm61-port: for FNX_27B (29 GB in 128 GB RAM, stored on 8 TB HDD), always read through the page cache so the
        // file stays warm in RAM and never triggers 4-thread O_DIRECT head seeks on the mechanical HDD
        const int df = FNX_27B ? -1 : ::open(g.shard_path(s).c_str(), O_RDONLY | O_DIRECT);
        if (df < 0 && !FNX_27B && errno != EINVAL) throw std::runtime_error("stager: cannot open " + g.shard_path(s) + ": " + std::strerror(errno));
        no_direct |= (df < 0 && !FNX_27B);
        direct_fd_.push_back(df);
    }
    if (no_direct) std::fprintf(stderr, "gguf: the model's file system refuses O_DIRECT: read through the page cache\n");
    groups_.resize(groups.size());
    for (size_t i = 0; i < groups.size(); ++i) {
        G& gr = groups_[i];
        gr.spec = std::move(groups[i]);
        gr.left = consumers;
        std::vector<const Tensor*> ts;
        for (const std::string& n : gr.spec.names) {
            const Tensor& t = g.t(n);
            if (t.shard < 0) throw std::runtime_error("stager: " + n + " is not in a file");
            ts.push_back(&t);
        }
        std::sort(ts.begin(), ts.end(), [](const Tensor* a, const Tensor* b) { return a->shard != b->shard ? a->shard < b->shard : a->offset < b->offset; });
        for (const Tensor* t : ts) {
            const uint64_t a = t->offset / PAGE * PAGE, e = (t->offset + t->nbytes + PAGE - 1) / PAGE * PAGE;
            Span* last = gr.spans.empty() ? nullptr : &gr.spans.back();
            if (last && last->shard == t->shard && a <= last->off + last->len + MERGE_GAP) {
                last->len = std::max(last->len, e - last->off);
            } else {
                if (last) gr.bytes += last->len;
                gr.spans.push_back(Span{t->shard, a, e - a, 0});
                gr.spans.back().at = gr.bytes;
            }
        }
        if (!gr.spans.empty()) gr.bytes += gr.spans.back().len;
        // each tensor's copy, its data an offset into the buffer until the buffer is there (read_group)
        for (const Tensor* t : ts) {
            const Span* sp = nullptr;
            for (const Span& s : gr.spans)
                if (s.shard == t->shard && t->offset >= s.off && t->offset + t->nbytes <= s.off + s.len) sp = &s;
            if (!sp) throw std::runtime_error("stager: " + t->name + " fell outside its group's ranges");
            Tensor c = *t;
            c.data = (const uint8_t*) (uintptr_t) (sp->at + (t->offset - sp->off));
            gr.t[t->name] = c;
        }
        buf_bytes_ = std::max(buf_bytes_, gr.bytes);
    }
    reader_ = std::thread([this] { run(); });
}

Stager::~Stager() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (error_.empty()) error_ = "stager: closed";
    }
    cv_.notify_all();
    reader_.join();
    for (G& gr : groups_) std::free(gr.buf);
    for (uint8_t* b : free_) std::free(b);
    for (int fd : plain_fd_) ::close(fd);
    for (int fd : direct_fd_)
        if (fd >= 0) ::close(fd);
}

void Stager::run() {
    for (G& gr : groups_) {
        uint8_t* buf = nullptr;
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [&] { return held_ < ahead_ || !error_.empty(); });
            if (!error_.empty()) return;
            ++held_;
            if (!free_.empty()) buf = free_.back(), free_.pop_back();
        }
        try {
            if (!buf) {
                void* p = nullptr;
                if (posix_memalign(&p, PAGE, std::max(buf_bytes_, PAGE)))
                    throw std::runtime_error("stager: " + std::to_string(buf_bytes_ >> 20) + " MiB for a group's buffer");
                buf = (uint8_t*) p;
            }
            gr.buf = buf;
            read_group(gr);
        } catch (const std::exception& e) {
            abort(e.what());
            return;
        }
        {
            std::lock_guard<std::mutex> lk(mu_);
            gr.ready = true;
        }
        cv_.notify_all();
    }
}

// the group's ranges in PIECE-sized pieces, READERS threads taking the next piece: one wholly in the page cache read
// through it (no disk), any other past it, so a cold model does not also fill the page cache (the loader keeps its
// own copies; Q8_0's file and copies together do not fit in RAM)
void Stager::read_group(G& gr) {
    const auto t0 = std::chrono::steady_clock::now();
    struct Piece { const Span* s; uint64_t k; };
    std::vector<Piece> pieces;
    for (const Span& s : gr.spans)
        for (uint64_t k = 0; k < s.len; k += PIECE) pieces.push_back(Piece{&s, k});
    std::atomic<size_t> next{0};
    std::vector<std::string> errs(READERS);
    std::vector<std::thread> th;
    for (int r = 0; r < READERS; ++r)
        th.emplace_back([&, r] {
            std::vector<unsigned char> inc(PIECE / PAGE);
            try {
                for (size_t i; (i = next.fetch_add(1)) < pieces.size();) {
                    const Span& s = *pieces[i].s;
                    const uint64_t fo = s.off + pieces[i].k, n = std::min(PIECE, s.len - pieces[i].k);
                    const uint64_t file = g_.shard_size(s.shard), valid = std::min(n, file - fo);   // the span's last page may pass the end
                    if (mincore((void*) (g_.shard_base(s.shard) + fo), valid, inc.data()) != 0)
                        throw std::runtime_error(std::string("stager: mincore: ") + std::strerror(errno));
                    bool cached = gr.spec.cached;
                    if (!cached) {
                        cached = true;
                        for (uint64_t p = 0; p < (valid + PAGE - 1) / PAGE; ++p) cached &= inc[p] & 1;
                    }
                    const int fd = cached || direct_fd_[(size_t) s.shard] < 0 ? plain_fd_[(size_t) s.shard] : direct_fd_[(size_t) s.shard];
                    uint8_t* dst = gr.buf + s.at + pieces[i].k;
                    uint64_t got = 0;
                    while (got < valid) {
                        // O_DIRECT reads whole pages: the last piece of a file asks past its end and gets what is there
                        const ssize_t k = ::pread(fd, dst + got, (size_t) (n - got), (off_t) (fo + got));
                        if (k < 0 && errno == EINTR) continue;
                        if (k <= 0)
                            throw std::runtime_error("stager: read of " + g_.shard_path(s.shard) + " at " + std::to_string(fo + got) + ": " +
                                                     (k < 0 ? std::strerror(errno) : "end of file"));
                        got += (uint64_t) k;
                    }
                    (cached ? cache_bytes_ : disk_bytes_) += valid;
                }
            } catch (const std::exception& e) {
                errs[(size_t) r] = e.what();
                next = pieces.size();
            }
        });
    for (auto& t : th) t.join();
    for (const std::string& e : errs)
        if (!e.empty()) throw std::runtime_error(e);
    for (auto& [name, t] : gr.t) t.data = gr.buf + (uintptr_t) t.data;
    read_s_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

const Tensor& Stager::t(int group, const std::string& name) {
    G& gr = groups_.at((size_t) group);
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait(lk, [&] { return gr.ready || !error_.empty(); });
    if (!error_.empty()) throw std::runtime_error(error_);
    const auto it = gr.t.find(name);
    if (it == gr.t.end()) throw std::runtime_error("stager: " + name + " is not in group " + std::to_string(group));
    return it->second;
}

void Stager::done(int group) {
    G& gr = groups_.at((size_t) group);
    {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&] { return gr.ready || !error_.empty(); });
        if (!error_.empty()) return;   // ending anyway: the buffers are freed by the destructor
        if (gr.left <= 0) throw std::runtime_error("stager: group " + std::to_string(group) + " given back twice");
        if (--gr.left) return;
        free_.push_back(gr.buf);
        gr.buf = nullptr;
        --held_;
    }
    cv_.notify_all();
}

void Stager::abort(const std::string& why) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (error_.empty()) error_ = why;
    }
    cv_.notify_all();
}

}  // namespace fnx
