//
// Production-path benchmark: builds the index the normal way (DARE picks the Configuration, TSMDP
// picks the per-region fanout) and measures memory plus point-query and range-query latency.
//
// This harness deliberately does NOT define PILOT_DIAGNOSTICS, so it compiles the index without the
// per-leaf instrumentation: no leaf_id field, no leaf_stats vector, and no shared-state write on
// the lookup path. Those cost 8 bytes per leaf and make an ordered-leaf lookup a data race under
// concurrent readers, so this is the configuration that would ship.
//
// Leaf layout: /home/... no -- the criterion is the same env knob as the pilot harnesses:
//   LEAF_WINDOW_THRESHOLD > 0 -> a leaf becomes ordered when its measured p99 rank-prediction
//                                error, in slots, is within the threshold
//   unset                     -> every leaf stays a hash leaf (baseline)
//
// Env: PROD_DATASET (default uden.data), PROD_LENGTH, PROD_QUERIES, PROD_REPEATS,
//      FIXED_CONF=<root>,<inner> overrides the agents' structure choice
// Range workload, two modes (both may be requested in one run):
//   PROD_RANGE=<s,...>      selectivity mode: the span is s*n in RANK units, so the result count
//                           is span+1 whatever the data looks like
//   PROD_RANGE_W=<W,...>    key-value mode: start a is a random key from the data, end is a+W in
//                           KEY units, so the result count is whatever the local density gives
//                           (PROD_RANGE_W_QUERIES, default 20000, sets the query count)
//

#include <iostream>
#include <random>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <x86intrin.h>
#include <torch/torch.h>
#include "../include/DEFINE.h"

#define using_small_network

#include "../index/include/Index.hpp"
#include "../include/DataSet.hpp"
#include "../index/include/experience.hpp"
#include "../index/include/Controller.hpp"

GlobalController controller;

// Leaf census by walking the tree the public way (children are ordered and the bitmap says which
// kind each slot holds). Lets this harness report structure without the pilot's counters.
template<class key_T, class value_T>
static void census(Hits::InnerNode<key_T, value_T> *node, long long &leaves,
                   long long &sum, long long &sum_sq) {
    for (int i = 0; i < node->capacity; ++i) {
        if (get_bitmap(node->bitmap_start(), i)) {
            census<key_T, value_T>(node->array[i].inner_node, leaves, sum, sum_sq);
        } else {
            long long size = node->array[i].data_node->size;
            ++leaves;
            sum += size;
            sum_sq += size * size;
        }
    }
}

// Leaves in key order with their key intervals and sizes. Children are equal-width and key-ordered,
// so a depth-first walk emits the leaves ordered by key, and each leaf's interval is its parent slot's
// sub_interval -- the very interval build_leaf_ saw. Used to label a query's leaf hot/cold.
template<class key_T, class value_T>
static void census_iv(Hits::InnerNode<key_T, value_T> *node,
                      std::vector<std::pair<double, double>> &ivs, std::vector<long long> &sizes) {
    for (int i = 0; i < node->capacity; ++i) {
        if (get_bitmap(node->bitmap_start(), i)) {
            census_iv<key_T, value_T>(node->array[i].inner_node, ivs, sizes);
        } else {
            ivs.push_back(node->sub_interval(i));
            sizes.push_back(node->array[i].data_node->size);
        }
    }
}

int main() {
    controller.load_in();
    e.seed(12345);// the policy must be reproducible across configurations

    std::string dataset_name = std::getenv("PROD_DATASET") ? std::getenv("PROD_DATASET") : "uden.data";
    long length = std::getenv("PROD_LENGTH") ? std::atol(std::getenv("PROD_LENGTH")) : 20000000L;
    long n_queries = std::getenv("PROD_QUERIES") ? std::atol(std::getenv("PROD_QUERIES")) : 40000000L;
    const char *range_env = std::getenv("PROD_RANGE");
    const char *rep_env = std::getenv("PROD_REPEATS");
    const int repeats = rep_env ? std::max(1, std::atoi(rep_env)) : 5;

    auto dataset = dataset_source::get_dataset<std::pair<KEY_TYPE, VALUE_TYPE>>(
            data_father_path + dataset_name);
    if ((long) dataset.size() > length) { dataset.resize(length); }
    std::sort(dataset.begin(), dataset.end(),
              [](const std::pair<KEY_TYPE, VALUE_TYPE> &a, const std::pair<KEY_TYPE, VALUE_TYPE> &b) {
                  return a.first < b.first;
              });
    const long n = (long) dataset.size();
    auto min_max = get_min_max<KEY_TYPE, VALUE_TYPE>(dataset.begin(), dataset.end());

    // structure: the agents by default, or a pinned Configuration for controlled runs
    Hits::Configuration conf;
    bool fixed_conf = false;
    double fixed_root = 0;
    double fixed_inner = 0;
    const char *fc_env = std::getenv("FIXED_CONF");
    if (fc_env && std::sscanf(fc_env, "%lf,%lf", &fixed_root, &fixed_inner) == 2) {
        conf.root_fan_out = float(fixed_root);
        for (auto &row: conf.fan_outs) {
            for (auto &v: row) { v = float(fixed_inner); }
        }
        fixed_conf = true;
    } else {
        auto pdf = get_pdf<KEY_TYPE, VALUE_TYPE>(dataset.begin(), dataset.end(),
                                                 min_max.first, min_max.second, BUCKET_SIZE);
        experience_t exp_chosen;
        std::copy(pdf.begin(), pdf.end(), exp_chosen.distribution);
        exp_chosen.data_size = float(n);
        conf = controller.get_best_action_GA(exp_chosen).conf;
    }

    // >>> workload model <<<
    // A rank sampler that sends PROD_WORK_HOT_Q of the queries into a contiguous hot rank interval
    // of width PROD_WORK_HOT_W (starting at PROD_WORK_HOT_POS), the rest uniform. The SAME sampler
    // fills workload_model (histogrammed by key, before bulk_load) and drives the point-query phase,
    // so the layout is built for exactly the workload it is measured on. hot_w <= 0 -> uniform, and
    // workload_model stays empty -> every leaf keeps default_density.
    const char *ww_env = std::getenv("PROD_WORK_HOT_W");
    const double hot_w = ww_env ? std::atof(ww_env) : -1.0;
    const double hot_q = std::getenv("PROD_WORK_HOT_Q") ? std::atof(std::getenv("PROD_WORK_HOT_Q")) : 0.8;
    const double hot_pos = std::getenv("PROD_WORK_HOT_POS") ? std::atof(std::getenv("PROD_WORK_HOT_POS")) : 0.4;
    auto skew_rank = [&](std::mt19937_64 &r) -> long {
        std::uniform_real_distribution<double> u(0.0, 1.0);
        if (u(r) < hot_q) {
            long a = (long) (hot_pos * double(n));
            long b = (long) ((hot_pos + hot_w) * double(n)) - 1;
            if (b <= a) { b = a; }
            std::uniform_int_distribution<long> d(a, b);
            return d(r);
        }
        std::uniform_int_distribution<long> d(0, n - 1);
        return d(r);
    };
    if (hot_w > 0) {
        const int WB = BUCKET_SIZE;
        std::vector<double> hist(WB, 0.0);
        const long samples = 4000000;
        std::mt19937_64 hrng(4242);
        const double span = min_max.second - min_max.first;
        for (long s = 0; s < samples; ++s) {
            long r = skew_rank(hrng);
            double p = double(WB) * (double(dataset[r].first) - min_max.first) / span;
            int b = std::min(WB - 1, std::max(0, int(p)));
            hist[b] += 1.0;
        }
        workload_model.lower = min_max.first;
        workload_model.upper = min_max.second;
        workload_model.bins = WB;
        workload_model.keys = double(n);
        workload_model.total = double(samples);
        workload_model.hist = hist;
        workload_stats = WorkloadStats();
        printf("workload: model=on hot_w=%g hot_q=%g hot_pos=%g samples=%ld bins=%d apply=%d cold_density=%g\n",
               hot_w, hot_q, hot_pos, samples, WB, workload_apply, workload_cold_density);
        fflush(stdout);
    }

    auto index = new Hits::Index<KEY_TYPE, VALUE_TYPE>(conf, min_max.first, min_max.second);
    auto t_build0 = std::chrono::high_resolution_clock::now();
    index->bulk_load(dataset.begin(), dataset.end());
    auto t_build1 = std::chrono::high_resolution_clock::now();
    const double build_ms = std::chrono::duration<double, std::milli>(t_build1 - t_build0).count();

    const double mem_bytes = index->memory_occupied();
    const double mem_ratio = mem_bytes / (double(n) * sizeof(std::pair<KEY_TYPE, VALUE_TYPE>));

    long long leaves = 0, leaf_sum = 0, leaf_sum_sq = 0;
    census<KEY_TYPE, VALUE_TYPE>(index->root, leaves, leaf_sum, leaf_sum_sq);

    printf("prod: dataset=%-16s n=%-9ld structure=%-6s root=%-6.0f leaves=%-7lld "
           "size_weighted_leaf=%-7.1f mem_MB=%-8.2f mem_ratio=%-7.4f build_ms=%.0f\n",
           dataset_name.c_str(), n, fixed_conf ? "fixed" : "policy", conf.root_fan_out, leaves,
           double(leaf_sum_sq) / double(std::max(1LL, leaf_sum)),
           mem_bytes / (1024.0 * 1024.0), mem_ratio, build_ms);

    if (workload_model.active()) {
        const double kappa = double(workload_stats.cold_keys) / double(std::max(1LL, workload_stats.all_keys));
        const double phi = (workload_stats.all_q > 0.0) ? workload_stats.cold_q / workload_stats.all_q : 0.0;
        printf("workload[stats]: leaves=%lld cold_leaves=%lld (%.1f%%) kappa=%.4f phi=%.4f\n",
               workload_stats.leaves, workload_stats.cold_leaves,
               100.0 * double(workload_stats.cold_leaves) / double(std::max(1LL, workload_stats.leaves)),
               kappa, phi);
        fflush(stdout);
    }

    std::mt19937_64 rng(1000);

    // ---- point queries ----
    {
        std::uniform_int_distribution<long> dist(0, n - 1);
        const long per_pass = std::max<long>(1, n_queries / repeats);
        VALUE_TYPE value;
        long errors = 0;
        std::vector<double> pass_ns;
        for (int rep = 0; rep < repeats; ++rep) {
            auto t0 = std::chrono::high_resolution_clock::now();
            for (long q = 0; q < per_pass; ++q) {
                long id = (hot_w > 0) ? skew_rank(rng) : dist(rng);
                if (!index->get(dataset[id].first, value) || value != dataset[id].second) { ++errors; }
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            pass_ns.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count() / double(per_pass));
        }
        std::vector<double> measured(pass_ns.begin() + 1, pass_ns.end());
        std::sort(measured.begin(), measured.end());
        printf("  point: queries=%-9ld ns=%.2f (med %.2f max %.2f, %zu passes) errors=%ld\n",
               per_pass * (long) measured.size(), measured.front(),
               measured[measured.size() / 2], measured.back(), measured.size(), errors);
        fflush(stdout);
    }

    // ---- point latency on ONE path: mixed stream, per-query rdtsc, binned by leaf class ----
    // A single interleaved run. Each query is timed individually (rdtsc) and binned by whether its
    // target leaf is hot or cold, so the reported total IS the (1-phi)*L_hot + phi*L_cold mixture by
    // construction -- total, hot and cold all come from the same run and reconcile. (The wall-clock
    // "point:" line above is a different path and is NOT comparable to these, as usual.) The rdtsc
    // pair adds a constant per query, so the absolute levels sit a little above "point:". The class
    // comes from the build-time leaf intervals (census) plus the very rule build_leaf_ used.
    if (workload_model.active()) {
        std::vector<std::pair<double, double>> ivs;
        std::vector<long long> szs;
        census_iv<KEY_TYPE, VALUE_TYPE>(index->root, ivs, szs);
        std::vector<double> hi;
        std::vector<char> cold;
        long long cold_leaves = 0;
        for (size_t i = 0; i < ivs.size(); ++i) {
            hi.push_back(ivs[i].second);
            char c = workload_leaf_is_cold(int(szs[i]), ivs[i].first, ivs[i].second) ? 1 : 0;
            cold.push_back(c);
            cold_leaves += c;
        }
        auto is_cold_key = [&](double key) -> bool {
            if (hi.empty()) { return false; }
            size_t i = size_t(std::lower_bound(hi.begin(), hi.end(), key) - hi.begin());
            if (i >= cold.size()) { i = cold.size() - 1; }
            return cold[i] != 0;
        };
        double ticks_per_ns = 1.0;
        {
            auto w0 = std::chrono::high_resolution_clock::now();
            uint64_t c0 = __rdtsc();
            volatile double acc = 0.0;
            for (long k = 0; k < 20000000; ++k) { acc += double(k); }
            uint64_t c1 = __rdtsc();
            auto w1 = std::chrono::high_resolution_clock::now();
            double ns = std::chrono::duration<double, std::nano>(w1 - w0).count();
            if (ns > 0) { ticks_per_ns = double(c1 - c0) / ns; }
            (void) acc;
        }
        std::mt19937_64 srng(1000);
        std::uniform_int_distribution<long> sdist(0, n - 1);
        const long sper = std::max<long>(1, n_queries / repeats);
        VALUE_TYPE v;
        long errors = 0, hot_total = 0, cold_total = 0;
        std::vector<double> tpass, hpass, cpass;
        for (int rep = 0; rep < repeats; ++rep) {
            uint64_t hot_c = 0, cold_c = 0;
            long hot_n = 0, cold_n = 0;
            for (long q = 0; q < sper; ++q) {
                long id = (hot_w > 0) ? skew_rank(srng) : sdist(srng);
                bool c = is_cold_key(double(dataset[id].first));
                uint64_t c0 = __rdtsc();
                bool ok = index->get(dataset[id].first, v) && (v == dataset[id].second);
                uint64_t c1 = __rdtsc();
                if (!ok) { ++errors; }
                if (c) { cold_c += c1 - c0; ++cold_n; } else { hot_c += c1 - c0; ++hot_n; }
            }
            hot_total += hot_n;
            cold_total += cold_n;
            if (hot_n + cold_n > 0) { tpass.push_back(double(hot_c + cold_c) / ticks_per_ns / double(hot_n + cold_n)); }
            if (hot_n > 0) { hpass.push_back(double(hot_c) / ticks_per_ns / double(hot_n)); }
            if (cold_n > 0) { cpass.push_back(double(cold_c) / ticks_per_ns / double(cold_n)); }
        }
        auto report = [](const char *tag, std::vector<double> &p) {
            if (p.size() > 1) { p.erase(p.begin()); }// drop warm-up pass
            if (p.empty()) { return; }
            std::sort(p.begin(), p.end());
            printf("  %s ns(min)=%.2f ns(med)=%.2f (n=%zu)\n", tag, p.front(), p[p.size() / 2], p.size());
        };
        printf("  byclass: leaves=%zu cold=%lld queries hot=%ld cold=%ld (%ld%%) errors=%ld ticks_per_ns=%.3f\n",
               ivs.size(), cold_leaves, hot_total, cold_total,
               (long) (100.0 * double(cold_total) / double(std::max(1L, hot_total + cold_total))),
               errors, ticks_per_ns);
        report("point-t  ", tpass);
        report("hot-leaf ", hpass);
        report("cold-leaf", cpass);
        fflush(stdout);
    }

    // ---- range queries: [lo,hi] spanning a fixed fraction s of the dataset.
    // PROD_RANGE takes a comma-separated list, so one build serves every selectivity: the
    // structure and the leaf layout do not depend on s.
    if (range_env != nullptr) {
        std::vector<double> sels;
        for (const char *p = range_env, *q = range_env; ; ++q) {
            if (*q == ',' || *q == '\0') {
                if (q > p) { sels.push_back(std::atof(std::string(p, q - p).c_str())); }
                if (*q == '\0') { break; }
                p = q + 1;
            }
        }
        auto by_key = [](const std::pair<KEY_TYPE, VALUE_TYPE> &a, KEY_TYPE v) { return a.first < v; };
        auto above = [](KEY_TYPE v, const std::pair<KEY_TYPE, VALUE_TYPE> &a) { return v < a.first; };
        for (double s: sels) {
            const long span = std::max(1L, (long) (s * double(n)));
            const long queries = std::max(200L, std::min(2000000L, 200000000L / span));
            std::uniform_int_distribution<long> dist(0, std::max<long>(0, n - span - 1));
            std::vector<long> starts(queries);
            for (long q = 0; q < queries; ++q) { starts[q] = dist(rng); }

            std::vector<std::pair<KEY_TYPE, VALUE_TYPE>> out;
            out.reserve((size_t) std::min<long>(2 * span + 16, 2000000L));
            long errors = 0;
            long long emitted = 0;
            long verified = 0;
            std::vector<double> pass_ns;
            for (int rep = 0; rep < repeats; ++rep) {
                auto t0 = std::chrono::high_resolution_clock::now();
                for (long q = 0; q < queries; ++q) {
                    const long st = starts[q];
                    index->range_query(dataset[st].first, dataset[st + span].first, out);
                    emitted += (long long) out.size();
                }
                auto t1 = std::chrono::high_resolution_clock::now();
                pass_ns.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count() / double(queries));
            }
            std::vector<double> measured(pass_ns.begin() + 1, pass_ns.end());
            std::sort(measured.begin(), measured.end());
            // correctness against the brute-force oracle (untimed)
            for (long q = 0; q < queries; ++q) {
                const long st = starts[q];
                const KEY_TYPE lo = dataset[st].first;
                const KEY_TYPE hi = dataset[st + span].first;
                index->range_query(lo, hi, out);
                auto a = std::lower_bound(dataset.begin(), dataset.end(), lo, by_key);
                auto b = std::upper_bound(dataset.begin(), dataset.end(), hi, above);
                if ((long) out.size() != (long) (b - a)) { ++errors; continue; }
                if (q < 200) {
                    std::sort(out.begin(), out.end(),
                              [](const std::pair<KEY_TYPE, VALUE_TYPE> &x, const std::pair<KEY_TYPE, VALUE_TYPE> &y) {
                                  return x.first < y.first;
                              });
                    if (!std::equal(out.begin(), out.end(), a)) { ++errors; }
                    ++verified;
                }
            }
            const double result_avg = double(emitted) / double(queries * repeats);
            printf("  range: s=%-9g span=%-7ld queries=%-7ld result_avg=%-8.1f ns=%.2f "
                   "(med %.2f max %.2f, %zu passes) ns_per_result=%.4f verified=%ld errors=%ld\n",
                   s, span, queries, result_avg, measured.front(),
                   measured[measured.size() / 2], measured.back(), measured.size(),
                   measured.front() / std::max(1.0, result_avg), verified, errors);
            fflush(stdout);
        }
    }

    // ---- range queries #2: fixed KEY-VALUE width. The start a is a random key drawn from the data
    // (uniform over ranks) and the end is a + W in key units, so the number of keys inside is
    // whatever the local density gives. This complements the selectivity mode above, which holds
    // the result count fixed. PROD_RANGE_W is a comma-separated list of widths in key units;
    // PROD_RANGE_W_QUERIES (default 20000) sets the query count, which cannot be derived from a
    // target emitted volume here because the emitted volume is data-dependent.
    const char *rw_env = std::getenv("PROD_RANGE_W");
    if (rw_env != nullptr) {
        std::vector<double> widths;
        for (const char *p = rw_env, *q = rw_env; ; ++q) {
            if (*q == ',' || *q == '\0') {
                if (q > p) { widths.push_back(std::atof(std::string(p, q - p).c_str())); }
                if (*q == '\0') { break; }
                p = q + 1;
            }
        }
        const long w_queries = std::getenv("PROD_RANGE_W_QUERIES")
                               ? std::atol(std::getenv("PROD_RANGE_W_QUERIES")) : 20000L;
        auto by_key = [](const std::pair<KEY_TYPE, VALUE_TYPE> &a, KEY_TYPE v) { return a.first < v; };
        auto above = [](KEY_TYPE v, const std::pair<KEY_TYPE, VALUE_TYPE> &a) { return v < a.first; };
        for (double W: widths) {
            std::uniform_int_distribution<long> dist(0, n - 1);
            std::vector<long> starts(w_queries);
            for (long q = 0; q < w_queries; ++q) { starts[q] = dist(rng); }

            std::vector<std::pair<KEY_TYPE, VALUE_TYPE>> out;
            out.reserve(1 << 20);
            std::vector<long> sizes(w_queries, 0);
            long errors = 0;
            long long emitted = 0;
            long verified = 0;
            std::vector<double> pass_ns;
            for (int rep = 0; rep < repeats; ++rep) {
                auto t0 = std::chrono::high_resolution_clock::now();
                for (long q = 0; q < w_queries; ++q) {
                    const KEY_TYPE lo = dataset[starts[q]].first;
                    index->range_query(lo, lo + W, out);
                    sizes[q] = (long) out.size();
                    emitted += (long long) out.size();
                }
                auto t1 = std::chrono::high_resolution_clock::now();
                pass_ns.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count() / double(w_queries));
            }
            std::vector<double> measured(pass_ns.begin() + 1, pass_ns.end());
            std::sort(measured.begin(), measured.end());
            // correctness (untimed): count for every query, multiset for the first 200
            long zeros = 0;
            for (long q = 0; q < w_queries; ++q) {
                const KEY_TYPE lo = dataset[starts[q]].first;
                const KEY_TYPE hi = lo + W;
                index->range_query(lo, hi, out);
                if (out.empty()) { ++zeros; }
                auto a = std::lower_bound(dataset.begin(), dataset.end(), lo, by_key);
                auto b = std::upper_bound(dataset.begin(), dataset.end(), hi, above);
                if ((long) out.size() != (long) (b - a)) { ++errors; continue; }
                if (q < 200) {
                    std::sort(out.begin(), out.end(),
                              [](const std::pair<KEY_TYPE, VALUE_TYPE> &x, const std::pair<KEY_TYPE, VALUE_TYPE> &y) {
                                  return x.first < y.first;
                              });
                    if (!std::equal(out.begin(), out.end(), a)) { ++errors; }
                    ++verified;
                }
            }
            auto sorted_sizes = sizes;
            std::sort(sorted_sizes.begin(), sorted_sizes.end());
            auto sq = [&](double p) {
                return sorted_sizes[std::min(sorted_sizes.size() - 1, (size_t) (p * sorted_sizes.size()))];
            };
            const double result_avg = double(emitted) / double(w_queries * repeats);
            printf("  rangeW: W=%-9g queries=%-7ld result min=%-6ld p50=%-7ld p90=%-8ld p99=%-8ld max=%-8ld "
                   "zero=%.1f%% avg=%.1f ns=%.2f (med %.2f max %.2f, %zu passes) ns_per_result=%.4f verified=%ld errors=%ld\n",
                   W, w_queries, sorted_sizes.front(), sq(0.50), sq(0.90), sq(0.99), sorted_sizes.back(),
                   100.0 * double(zeros) / double(w_queries), result_avg,
                   measured.front(), measured[measured.size() / 2], measured.back(), measured.size(),
                   measured.front() / std::max(1.0, result_avg), verified, errors);
            fflush(stdout);
        }
    }

    delete index;
    return 0;
}
