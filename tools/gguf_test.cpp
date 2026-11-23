// Host-only test of the GGUF reader's header checks: writes tiny GGUF files, one valid and several malformed, and
// expects the valid one to load and each malformed one to be refused with a message naming the tensor; and the rotary
// scaling a model's keys come to (model_rope), each llama.cpp reading of them followed or refused.
// make bin/gguf_test && bin/gguf_test [DIR [MODEL.gguf ...]]   (DIR for the files, default /tmp; each MODEL, a real
// model's first shard, must load too, with a rotary sFNX follows: only its headers are read)
#include "../src/gguf.hpp"
#include "../src/rope_gguf.hpp"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// metadata entries as GGUF lays them out: a string, a uint32 and a float32 key (types 8, 4 and 6)
struct Kv {
    std::vector<uint8_t> b;
    uint64_t n = 0;
    void put(const void* p, size_t len) { b.insert(b.end(), (const uint8_t*) p, (const uint8_t*) p + len); }
    void key(const std::string& k, uint32_t type) {
        const uint64_t len = k.size();
        put(&len, 8);
        put(k.data(), k.size());
        put(&type, 4);
        ++n;
    }
    void str(const std::string& k, const std::string& v) {
        key(k, 8);
        const uint64_t len = v.size();
        put(&len, 8);
        put(v.data(), v.size());
    }
    void u32(const std::string& k, uint32_t v) { key(k, 4); put(&v, 4); }
    void f32(const std::string& k, float v) { key(k, 6); put(&v, 4); }
};

// one F32 tensor "w" with the given n_dims and sizes, its data at offset 0 of the data section, data_bytes of data
std::vector<uint8_t> make(uint32_t n_dims, const std::vector<uint64_t>& ne, uint64_t data_bytes, uint64_t offset = 0, const Kv& kv = {}) {
    std::vector<uint8_t> b;
    auto put = [&](const void* p, size_t n) { b.insert(b.end(), (const uint8_t*) p, (const uint8_t*) p + n); };
    auto u32 = [&](uint32_t v) { put(&v, 4); };
    auto u64 = [&](uint64_t v) { put(&v, 8); };
    u32(0x46554747u);   // "GGUF"
    u32(3);             // version
    u64(1);             // tensors
    u64(kv.n);          // metadata keys
    put(kv.b.data(), kv.b.size());
    const std::string name = "w";
    u64(name.size());
    put(name.data(), name.size());
    u32(n_dims);
    for (uint64_t v : ne) u64(v);
    u32(0);             // F32
    u64(offset);
    while (b.size() % 32) b.push_back(0);
    b.resize(b.size() + data_bytes, 0);
    return b;
}

std::string write(const std::string& dir, const std::string& tag, const std::vector<uint8_t>& b) {
    const std::string path = dir + "/fnx_gguf_test_" + tag + ".gguf";
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f || std::fwrite(b.data(), 1, b.size(), f) != b.size() || std::fclose(f) != 0)
        throw std::runtime_error("cannot write " + path);
    return path;
}

struct TensorSpec {
    std::string name;
    uint32_t type;
    std::vector<uint64_t> ne;
    uint8_t fill;
};

std::vector<uint8_t> make_multi(const std::vector<TensorSpec>& specs, const Kv& kv = {}) {
    std::vector<uint8_t> b;
    auto put = [&](const void* p, size_t n) { b.insert(b.end(), (const uint8_t*) p, (const uint8_t*) p + n); };
    auto u32 = [&](uint32_t v) { put(&v, 4); };
    auto u64 = [&](uint64_t v) { put(&v, 8); };
    u32(0x46554747u);
    u32(3);
    u64(specs.size());
    u64(kv.n);
    put(kv.b.data(), kv.b.size());
    std::vector<uint64_t> sizes(specs.size());
    std::vector<uint64_t> offs(specs.size());
    uint64_t cur = 0;
    for (size_t i = 0; i < specs.size(); ++i) {
        const fnx::TypeInfo ti = fnx::type_info(specs[i].type);
        uint64_t nel = 1;
        for (uint64_t d : specs[i].ne) nel *= d;
        sizes[i] = (nel / ti.block) * ti.bytes;
        cur = (cur + 31ull) & ~31ull;
        offs[i] = cur;
        cur += sizes[i];
    }
    for (size_t i = 0; i < specs.size(); ++i) {
        u64(specs[i].name.size());
        put(specs[i].name.data(), specs[i].name.size());
        u32((uint32_t) specs[i].ne.size());
        for (uint64_t d : specs[i].ne) u64(d);
        u32(specs[i].type);
        u64(offs[i]);
    }
    while (b.size() % 32) b.push_back(0);
    const size_t data0 = b.size();
    b.resize(data0 + (size_t) cur, 0);
    for (size_t i = 0; i < specs.size(); ++i)
        std::memset(b.data() + data0 + offs[i], specs[i].fill, (size_t) sizes[i]);
    return b;
}

std::string write_exact(const std::string& path, const std::vector<uint8_t>& b) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f || std::fwrite(b.data(), 1, b.size(), f) != b.size() || std::fclose(f) != 0)
        throw std::runtime_error("cannot write " + path);
    return path;
}

bool all_bytes(const uint8_t* p, size_t n, uint8_t v) {
    for (size_t i = 0; i < n; ++i)
        if (p[i] != v) return false;
    return true;
}

int failures = 0;

void expect_refused(const std::string& dir, const std::string& tag, const std::vector<uint8_t>& b, const char* want) {
    const std::string path = write(dir, tag, b);
    try {
        fnx::Gguf g(path, false);
        std::printf("FAIL %s: loaded\n", tag.c_str());
        ++failures;
    } catch (const std::runtime_error& e) {
        const bool ok = std::strstr(e.what(), "tensor w") && std::strstr(e.what(), want);
        std::printf("%s %s: %s\n", ok ? "ok  " : "FAIL", tag.c_str(), e.what());
        failures += !ok;
    }
    std::remove(path.c_str());
}

}  // namespace

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : "/tmp";

    {
        const std::string path = write(dir, "valid", make(2, {8, 3}, 8 * 3 * 4));
        try {
            fnx::Gguf g(path, false);
            const fnx::Tensor& t = g.t("w");
            const bool ok = t.n_dims == 2 && t.ne[0] == 8 && t.ne[1] == 3 && t.ne[2] == 1 && t.ne[3] == 1 &&
                            t.nbytes == 96 && t.row_bytes() == 32;
            std::printf("%s valid: n_dims %d, ne %lld x %lld, %zu bytes\n", ok ? "ok  " : "FAIL", t.n_dims,
                        (long long) t.ne[0], (long long) t.ne[1], t.nbytes);
            failures += !ok;
        } catch (const std::runtime_error& e) {
            std::printf("FAIL valid: %s\n", e.what());
            ++failures;
        }
        std::remove(path.c_str());
    }

    {   // a model's YaRN keys as the loader reads them (Model, choose_rope): a string type, numbers in either width
        Kv kv;
        kv.str("a.rope.scaling.type", "yarn");
        kv.f32("a.rope.scaling.factor", 4.0f);
        kv.u32("a.rope.scaling.original_context_length", 32768);
        const std::string path = write(dir, "keys", make(2, {8, 3}, 8 * 3 * 4, 0, kv));
        try {
            fnx::Gguf g(path, false);
            const bool ok = g.has_key("a.rope.scaling.type") && g.s("a.rope.scaling.type") == "yarn" && g.num("a.rope.scaling.factor") == 4.0 &&
                            g.num("a.rope.scaling.original_context_length") == 32768.0 && g.i("a.rope.scaling.original_context_length") == 32768 &&
                            !g.has_key("a.rope.scaling.alpha");
            std::printf("%s keys: string, float and integer metadata read back\n", ok ? "ok  " : "FAIL");
            failures += !ok;
        } catch (const std::runtime_error& e) {
            std::printf("FAIL keys: %s\n", e.what());
            ++failures;
        }
        std::remove(path.c_str());
    }

    {   // model_rope: what llama.cpp makes of the keys, followed (factor, orig) or refused (the message holds `refused`)
        struct Case {
            const char* tag;
            void (*keys)(Kv&);
            double factor;
            int orig;
            const char* refused;
        };
        const Case cases[] = {
            {"rope_none_at_all", [](Kv&) {}, 0, 0, nullptr},
            {"rope_yarn", [](Kv& k) { k.str("a.rope.scaling.type", "yarn"); k.f32("a.rope.scaling.factor", 4); k.u32("a.rope.scaling.original_context_length", 32768); }, 4, 32768, nullptr},
            {"rope_yarn_no_orig", [](Kv& k) { k.str("a.rope.scaling.type", "yarn"); k.f32("a.rope.scaling.factor", 2); }, 2, 0, nullptr},
            {"rope_yarn_factor_1", [](Kv& k) { k.str("a.rope.scaling.type", "yarn"); k.f32("a.rope.scaling.factor", 1); }, 0, 0, nullptr},
            {"rope_none_with_factor", [](Kv& k) { k.str("a.rope.scaling.type", "none"); k.f32("a.rope.scaling.factor", 4); }, 0, 0, nullptr},
            {"rope_linear_factor_1", [](Kv& k) { k.str("a.rope.scaling.type", "linear"); k.f32("a.rope.scaling.factor", 1); }, 0, 0, nullptr},
            {"rope_yarn_default_betas", [](Kv& k) { k.str("a.rope.scaling.type", "yarn"); k.f32("a.rope.scaling.factor", 2); k.f32("a.rope.scaling.yarn_beta_fast", 32); }, 2, 0, nullptr},
            // no type is linear to llama.cpp, so a bare factor scales linearly there
            {"rope_bare_factor", [](Kv& k) { k.f32("a.rope.scaling.factor", 4); }, 0, 0, "llama.cpp takes that as linear"},
            {"rope_bare_scale_linear", [](Kv& k) { k.f32("a.rope.scale_linear", 2); }, 0, 0, "llama.cpp takes that as linear"},
            {"rope_linear_4", [](Kv& k) { k.str("a.rope.scaling.type", "linear"); k.f32("a.rope.scaling.factor", 4); }, 0, 0, "follows yarn only"},
            {"rope_longrope", [](Kv& k) { k.str("a.rope.scaling.type", "longrope"); }, 0, 0, "rope scaling is longrope"},
            {"rope_compress", [](Kv& k) { k.str("a.rope.scaling.type", "yarn"); k.f32("a.rope.scaling.factor", 0.5f); }, 0, 0, "compress"},
            {"rope_beta_fast_16", [](Kv& k) { k.str("a.rope.scaling.type", "yarn"); k.f32("a.rope.scaling.factor", 2); k.f32("a.rope.scaling.yarn_beta_fast", 16); }, 0, 0, "yarn_beta_fast"},
            {"rope_log_mul", [](Kv& k) { k.f32("a.rope.scaling.yarn_log_multiplier", 0.1f); }, 0, 0, "yarn_log_multiplier"},
            {"rope_attn_factor", [](Kv& k) { k.f32("a.rope.scaling.attn_factor", 0.5f); }, 0, 0, "attn_factor"},
            {"rope_section_order", [](Kv& k) { k.str("a.rope.section_order", "tzyx"); }, 0, 0, "section_order"},
        };
        for (const Case& c : cases) {
            Kv kv;
            c.keys(kv);
            const std::string path = write(dir, c.tag, make(2, {8, 3}, 8 * 3 * 4, 0, kv));
            std::string got;
            bool ok;
            try {
                fnx::Gguf g(path, false);
                const fnx::ModelRope r = fnx::model_rope(g, "a.");
                got = "x" + std::to_string(r.yarn_factor) + " from " + std::to_string(r.yarn_orig);
                ok = !c.refused && r.yarn_factor == c.factor && r.yarn_orig == c.orig;
            } catch (const std::runtime_error& e) {
                got = e.what();
                ok = c.refused && std::strstr(e.what(), c.refused);
            }
            std::printf("%s %s: %s\n", ok ? "ok  " : "FAIL", c.tag, got.c_str());
            failures += !ok;
            std::remove(path.c_str());
        }
    }

    expect_refused(dir, "dims0", make(0, {}, 64), "0 dimensions");
    expect_refused(dir, "dims5", make(5, {1, 1, 1, 1, 1}, 64), "5 dimensions");
    expect_refused(dir, "dims_huge", make(0xffffffffu, {1}, 64), "4294967295 dimensions");
    expect_refused(dir, "ne_zero", make(2, {8, 0}, 64), "size 0 in dimension 1");
    expect_refused(dir, "ne_negative", make(1, {~0ull}, 64), "in dimension 0");
    // 2^32 x 2^32 elements: 2^64 wraps to 0, which would pass the end check without the overflow test
    expect_refused(dir, "count_overflow", make(2, {1ull << 32, 1ull << 32}, 64), "element count overflows");
    // 2^62 elements of 4 bytes: the count fits, the byte count does not
    expect_refused(dir, "bytes_overflow", make(1, {1ull << 62}, 64), "runs past the end");
    expect_refused(dir, "past_end", make(2, {8, 3}, 95), "runs past the end");
    expect_refused(dir, "offset_wraps", make(1, {8}, 64, ~0ull - 8), "runs past the end");

    {   // FNX_ENGRAM_GGUF: missing final split shard skipped when external engram supplies per_layer_token_embd.weight
        const std::string base1 = dir + "/fnx_engram_base-00001-of-00002.gguf";
        const std::string base_miss = dir + "/fnx_engram_mid-00001-of-00003.gguf";
        const std::string base_full = dir + "/fnx_engram_full.gguf";
        const std::string ext_ok = dir + "/fnx_engram_ext_ok.gguf";
        const std::string ext_bad_name = dir + "/fnx_engram_ext_noname.gguf";
        const std::string ext_bad_shape = dir + "/fnx_engram_ext_shape.gguf";
        write_exact(base1, make_multi({{"token_embd.weight", fnx::T_Q8_0, {32, 4}, 0x21}}));
        write_exact(base_miss, make_multi({{"token_embd.weight", fnx::T_Q8_0, {32, 4}, 0x21}}));
        write_exact(base_full, make_multi({{"token_embd.weight", fnx::T_Q8_0, {32, 4}, 0x21},
                                           {"per_layer_token_embd.weight", fnx::T_IQ4_NL, {160, 4}, 0x11}}));
        write_exact(ext_ok, make_multi({{"per_layer_token_embd.weight", fnx::T_Q8_0, {160, 4}, 0x5a}}));
        write_exact(ext_bad_name, make_multi({{"other.weight", fnx::T_Q8_0, {160, 4}, 0x5a}}));
        write_exact(ext_bad_shape, make_multi({{"per_layer_token_embd.weight", fnx::T_Q8_0, {160, 8}, 0x5a}}));

        // 1. Missing final shard without engram must fail
        {
            bool ok = false;
            try { fnx::Gguf g(base1, false); } catch (const std::runtime_error& e) { ok = std::strstr(e.what(), "cannot open") != nullptr; }
            std::printf("%s engram_missing_tail_no_env: refused missing shard\n", ok ? "ok  " : "FAIL");
            failures += !ok;
        }
        // 2. Missing final shard WITH engram_gguf succeeds and supplies exact tensor & shard path
        {
            fnx::GgufOpts opts;
            opts.engram_gguf = ext_ok;
            bool ok = false;
            try {
                fnx::Gguf g(base1, opts);
                const fnx::Tensor& pt = g.t("per_layer_token_embd.weight");
                ok = g.has_engram_override() && pt.type == fnx::T_Q8_0 && pt.ne[0] == 160 && pt.ne[1] == 4 &&
                     g.shard_path(pt.shard) == ext_ok && all_bytes(pt.data, pt.nbytes, 0x5a);
            } catch (const std::runtime_error& e) { std::printf("  err: %s\n", e.what()); }
            std::printf("%s engram_missing_tail_override: loaded missing-tail model with external PLE\n", ok ? "ok  " : "FAIL");
            failures += !ok;
        }
        // 3. Missing non-final shard (00002-of-00003) still refused even with engram_gguf
        {
            fnx::GgufOpts opts;
            opts.engram_gguf = ext_ok;
            bool ok = false;
            try { fnx::Gguf g(base_miss, opts); } catch (const std::runtime_error& e) { ok = std::strstr(e.what(), "cannot open") != nullptr; }
            std::printf("%s engram_missing_nonfinal: refused missing middle shard\n", ok ? "ok  " : "FAIL");
            failures += !ok;
        }
        // 4. Engram shard missing per_layer_token_embd.weight refused
        {
            fnx::GgufOpts opts;
            opts.engram_gguf = ext_bad_name;
            bool ok = false;
            try { fnx::Gguf g(base1, opts); } catch (const std::runtime_error& e) { ok = std::strstr(e.what(), "per_layer_token_embd.weight") != nullptr; }
            std::printf("%s engram_missing_tensor: refused shard without PLE tensor\n", ok ? "ok  " : "FAIL");
            failures += !ok;
        }
        // 5. Override existing PLE tensor (Q8_0 replacing IQ4_NL, matching shape)
        {
            fnx::GgufOpts opts;
            opts.engram_gguf = ext_ok;
            bool ok = false;
            try {
                fnx::Gguf g(base_full, opts);
                const fnx::Tensor& pt = g.t("per_layer_token_embd.weight");
                ok = g.has_engram_override() && pt.type == fnx::T_Q8_0 && g.shard_path(pt.shard) == ext_ok &&
                     all_bytes(pt.data, pt.nbytes, 0x5a);
            } catch (const std::runtime_error& e) { std::printf("  err: %s\n", e.what()); }
            std::printf("%s engram_replace_existing: Q8_0 override replaced IQ4_NL\n", ok ? "ok  " : "FAIL");
            failures += !ok;
        }
        // 6. Shape mismatch refused
        {
            fnx::GgufOpts opts;
            opts.engram_gguf = ext_bad_shape;
            bool ok = false;
            try { fnx::Gguf g(base_full, opts); } catch (const std::runtime_error& e) { ok = std::strstr(e.what(), "does not match base") != nullptr; }
            std::printf("%s engram_shape_mismatch: refused mismatched PLE shape\n", ok ? "ok  " : "FAIL");
            failures += !ok;
        }
        for (const std::string& p : {base1, base_miss, base_full, ext_ok, ext_bad_name, ext_bad_shape}) std::remove(p.c_str());
    }

    {   // FNX_HYBRID_Q6_GGUF + FNX_HYBRID_MAP=q4q6q8: cross-shard splicing & multi-shard Stager verification
        const std::string bpath = dir + "/fnx_hyb_base.gguf";
        const std::string q6_1 = dir + "/fnx_hyb_q6-00001-of-00002.gguf";
        const std::string q6_2 = dir + "/fnx_hyb_q6-00002-of-00002.gguf";
        std::vector<TensorSpec> base_specs, q6_s1, q6_s2;
        for (int l = 0; l < 4; ++l) {
            const std::string p = "blk." + std::to_string(l) + ".";
            const uint32_t bgu = (l == 2) ? fnx::T_Q5_K : fnx::T_Q4_K;
            const uint32_t bdn = (l == 2) ? fnx::T_Q8_0 : fnx::T_Q5_1;
            base_specs.push_back({p + "ffn_gate_exps.weight", bgu, {256, 32, 2}, (uint8_t) (0x10 + l)});
            base_specs.push_back({p + "ffn_up_exps.weight", bgu, {256, 32, 2}, (uint8_t) (0x20 + l)});
            base_specs.push_back({p + "ffn_down_exps.weight", bdn, {32, 256, 2}, (uint8_t) (0x30 + l)});
            auto& dst = (l < 2) ? q6_s1 : q6_s2;
            const uint32_t sgu = (l == 2) ? fnx::T_Q8_0 : fnx::T_Q6_K;
            dst.push_back({p + "ffn_gate_exps.weight", sgu, {256, 32, 2}, (uint8_t) (0x60 + l)});
            dst.push_back({p + "ffn_up_exps.weight", sgu, {256, 32, 2}, (uint8_t) (0x70 + l)});
            dst.push_back({p + "ffn_down_exps.weight", fnx::T_Q8_0, {32, 256, 2}, (uint8_t) (0x80 + l)});
        }
        write_exact(bpath, make_multi(base_specs));
        write_exact(q6_1, make_multi(q6_s1));
        write_exact(q6_2, make_multi(q6_s2));

        {
            fnx::GgufOpts opts;
            opts.hybrid_q6_gguf = q6_1;
            bool ok = false;
            try {
                fnx::Gguf g(bpath, opts);
                auto chk = [&](const std::string& name, uint32_t tp, uint8_t fill) {
                    const fnx::Tensor& t = g.t(name);
                    return t.type == tp && all_bytes(t.data, t.nbytes, fill);
                };
                ok = g.has_hybrid_splice() && g.hybrid_map() == "q4q6q8" &&
                     chk("blk.0.ffn_gate_exps.weight", fnx::T_Q6_K, 0x60) &&
                     chk("blk.0.ffn_up_exps.weight", fnx::T_Q6_K, 0x70) &&
                     chk("blk.0.ffn_down_exps.weight", fnx::T_Q8_0, 0x80) &&
                     chk("blk.1.ffn_gate_exps.weight", fnx::T_Q6_K, 0x61) &&
                     chk("blk.1.ffn_down_exps.weight", fnx::T_Q8_0, 0x81) &&
                     chk("blk.2.ffn_gate_exps.weight", fnx::T_Q8_0, 0x62) &&
                     chk("blk.2.ffn_up_exps.weight", fnx::T_Q8_0, 0x72) &&
                     chk("blk.2.ffn_down_exps.weight", fnx::T_Q8_0, 0x82) &&
                     // Interior layer L03: gate/up remain base Q4_K, down spliced to Q8_0
                     chk("blk.3.ffn_gate_exps.weight", fnx::T_Q4_K, 0x13) &&
                     chk("blk.3.ffn_up_exps.weight", fnx::T_Q4_K, 0x23) &&
                     chk("blk.3.ffn_down_exps.weight", fnx::T_Q8_0, 0x83);
                // Verify Stager reads cross-shard L03 group (base Q4 gate/up + Q6 shard2 down) byte-identically
                fnx::Stager st(g, {fnx::Stager::Group{{"blk.3.ffn_gate_exps.weight", "blk.3.ffn_down_exps.weight"}, false}}, 1, 1);
                const fnx::Tensor& sg = st.t(0, "blk.3.ffn_gate_exps.weight");
                const fnx::Tensor& sd = st.t(0, "blk.3.ffn_down_exps.weight");
                ok = ok && all_bytes(sg.data, sg.nbytes, 0x13) && all_bytes(sd.data, sd.nbytes, 0x83);
                st.done(0);
            } catch (const std::runtime_error& e) { std::printf("  err: %s\n", e.what()); }
            std::printf("%s hybrid_q4q6q8_splice_and_stager: L00/L01=Q6+Q8, L02=Q8+Q8, L03=Q4+Q8 + Stager verified\n",
                        ok ? "ok  " : "FAIL");
            failures += !ok;
        }
        {
            fnx::GgufOpts opts;
            opts.hybrid_q6_gguf = q6_1;
            opts.hybrid_map = "bogus";
            bool ok = false;
            try { fnx::Gguf g(bpath, opts); } catch (const std::runtime_error& e) { ok = std::strstr(e.what(), "unsupported FNX_HYBRID_MAP") != nullptr; }
            std::printf("%s hybrid_bad_map: refused bogus map\n", ok ? "ok  " : "FAIL");
            failures += !ok;
        }
        for (const std::string& p : {bpath, q6_1, q6_2}) std::remove(p.c_str());
    }

    for (int a = 2; a < argc; ++a) {
        try {
            fnx::Gguf g(argv[a]);
            const fnx::ModelRope r = fnx::model_rope(g, g.s("general.architecture") + ".");
            std::printf("ok   %s: loads, rotary %s\n", argv[a], r.yarn_factor ? ("its own YaRN x" + std::to_string(r.yarn_factor)).c_str() : "unscaled");
        } catch (const std::runtime_error& e) {
            std::printf("FAIL %s: %s\n", argv[a], e.what());
            ++failures;
        }
    }

    std::printf(failures ? "%d FAILED\n" : "all passed\n", failures);
    return failures ? 1 : 0;
}
