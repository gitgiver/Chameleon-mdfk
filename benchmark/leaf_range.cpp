//
// Leaf-type-conditional range benchmark. Derived from prod_bench.cpp, which it leaves untouched.
// Builds the index the normal way, then takes a census of which leaves are ordered vs hash and
// measures range queries that fall ENTIRELY inside ordered leaves and ENTIRELY inside hash leaves
// (gated by PROD_LEAF_RANGE; PROD_LEAF_RANGE_N sets the scan length in keys, PROD_LEAF_RANGE_Q the
// query count). The point and rank/key-width range measurements from prod_bench still run first.
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
#include <functional>
#include <algorithm>
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
                long id = dist(rng);
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

    // ---- leaf-type-conditional range queries ----
    // Census the leaves in key order (their traversal order: children are stored by key interval),
    // then measure ranges that fall entirely inside ordered leaves vs entirely inside hash leaves.
    // Meaningful only on a mixed index (LEAF_COST_DECISION converting part of the leaves); an
    // all-ordered or all-hash index leaves one group empty. Off unless PROD_LEAF_RANGE is set.
    if (std::getenv("PROD_LEAF_RANGE") != nullptr) {
        std::vector<std::pair<long long, int>> leafinfo;//(size, ordered), in key order
        std::function<void(Hits::InnerNode<KEY_TYPE, VALUE_TYPE> *)> walk =
                [&](Hits::InnerNode<KEY_TYPE, VALUE_TYPE> *node) {
                    for (int i = 0; i < node->capacity; ++i) {
                        if (get_bitmap(node->bitmap_start(), i)) {
                            walk(node->array[i].inner_node);
                        } else {
                            auto leaf = node->array[i].data_node;
                            leafinfo.push_back({(long long) leaf->size, leaf->ordered});
                        }
                    }
                };
        walk(index->root);

        struct Zone { long long rank_lo, rank_hi; int ordered; };
        std::vector<Zone> zones;
        long long cum = 0;
        for (auto &li: leafinfo) {
            if (li.first == 0) { continue; }//empty leaves hold no keys and add no rank
            if (!zones.empty() && zones.back().ordered == li.second) {
                zones.back().rank_hi += li.first;
            } else {
                zones.push_back({cum, cum + li.first, li.second});
            }
            cum += li.first;
        }
        long long ordered_keys = 0, hash_keys = 0;
        for (auto &z: zones) { (z.ordered ? ordered_keys : hash_keys) += z.rank_hi - z.rank_lo; }

        const long N = std::getenv("PROD_LEAF_RANGE_N") ? std::atol(std::getenv("PROD_LEAF_RANGE_N")) : 100L;
        const long Q = std::getenv("PROD_LEAF_RANGE_Q") ? std::atol(std::getenv("PROD_LEAF_RANGE_Q")) : 100000L;
        auto by_key = [](const std::pair<KEY_TYPE, VALUE_TYPE> &a, KEY_TYPE v) { return a.first < v; };
        auto above = [](KEY_TYPE v, const std::pair<KEY_TYPE, VALUE_TYPE> &a) { return v < a.first; };
        printf("leafrange: leaves=%zu zones=%zu ordered_keys=%lld hash_keys=%lld (%.1f%% ordered) N=%ld\n",
               leafinfo.size(), zones.size(), ordered_keys, hash_keys,
               100.0 * double(ordered_keys) / double(std::max(1LL, ordered_keys + hash_keys)), N);
        fflush(stdout);

        // shared measurement over a fixed set of start ranks
        auto measure = [&](std::vector<long> &starts, const char *label) {
            std::vector<std::pair<KEY_TYPE, VALUE_TYPE>> out;
            out.reserve(1 << 20);
            long errors = 0, verified = 0;
            long long emitted = 0;
            std::vector<double> pass_avg;
            std::vector<long long> pass_total;
            for (int rep = 0; rep < repeats; ++rep) {
                auto t0 = std::chrono::high_resolution_clock::now();
                long long pass_emit = 0;
                for (long q = 0; q < Q; ++q) {
                    long st = starts[q];
                    index->range_query(dataset[st].first, dataset[st + N].first, out);
                    pass_emit += (long long) out.size();
                }
                auto t1 = std::chrono::high_resolution_clock::now();
                double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
                pass_avg.push_back(ns / double(Q));
                pass_total.push_back((long long) ns);
                emitted = pass_emit;
            }
            for (long q = 0; q < Q; ++q) {// correctness, untimed
                long st = starts[q];
                const KEY_TYPE lo = dataset[st].first, hi = dataset[st + N].first;
                index->range_query(lo, hi, out);
                auto a = std::lower_bound(dataset.begin(), dataset.end(), lo, by_key);
                auto b = std::upper_bound(dataset.begin(), dataset.end(), hi, above);
                if ((long) out.size() != (long) (b - a)) { ++errors; continue; }
                if (q < 200) {
                    std::sort(out.begin(), out.end(),
                              [](const std::pair<KEY_TYPE, VALUE_TYPE> &x, const std::pair<KEY_TYPE, VALUE_TYPE> &y) {
                                  return x.first < y.first; });
                    if (!std::equal(out.begin(), out.end(), a)) { ++errors; }
                    ++verified;
                }
            }
            size_t best = 0;
            for (size_t i = 1; i < pass_avg.size(); ++i) { if (pass_avg[i] < pass_avg[best]) { best = i; } }
            printf("  leafrange[%-11s]: queries=%-8ld total_ns=%-14lld avg_ns=%-9.2f result_avg=%.1f verified=%ld errors=%ld\n",
                   label, Q, pass_total[best], pass_avg[best], double(emitted) / double(Q), verified, errors);
            fflush(stdout);
        };

        // a range entirely inside same-type leaves (ordered or hash)
        auto run_zone_group = [&](int want_ordered, const char *label) {
            std::vector<size_t> zi;
            std::vector<long long> cumpfx;
            long long total = 0;
            for (size_t z = 0; z < zones.size(); ++z) {
                if (zones[z].ordered != want_ordered) { continue; }
                long long len = zones[z].rank_hi - zones[z].rank_lo;
                if (len > N) { zi.push_back(z); total += len - N; cumpfx.push_back(total); }
            }
            if (zi.empty()) { printf("  leafrange[%s]: no eligible zone\n", label); fflush(stdout); return; }
            std::uniform_int_distribution<long long> pick(0, total - 1);
            std::vector<long> starts(Q);
            for (long q = 0; q < Q; ++q) {
                long long t = pick(rng);
                size_t k = std::lower_bound(cumpfx.begin(), cumpfx.end(), t + 1) - cumpfx.begin();
                long long prev = (k == 0) ? 0 : cumpfx[k - 1];
                starts[q] = (long) (zones[zi[k]].rank_lo + (t - prev));
            }
            measure(starts, label);
        };

        // a range that straddles an ordered/hash boundary: ~N/2 keys on each side
        auto run_half_half = [&](const char *label) {
            const long long H = N / 2;
            std::vector<long long> anchors;
            for (size_t z = 0; z + 1 < zones.size(); ++z) {
                if (zones[z].ordered == zones[z + 1].ordered) { continue; }
                long long b = zones[z].rank_hi;// boundary rank: end of zone z, start of zone z+1
                long long start = b - H;
                long long end = start + N;
                if (start >= zones[z].rank_lo && end < zones[z + 1].rank_hi) { anchors.push_back(start); }
            }
            if (anchors.empty()) { printf("  leafrange[%s]: no straddling boundary\n", label); fflush(stdout); return; }
            printf("  half-half anchors=%zu first=%lld last=%lld\n", anchors.size(), anchors.front(), anchors.back());
            fflush(stdout);
            std::uniform_int_distribution<size_t> pick(0, anchors.size() - 1);
            std::vector<long> starts(Q);
            for (long q = 0; q < Q; ++q) { starts[q] = (long) anchors[pick(rng)]; }
            measure(starts, label);
        };

        run_zone_group(1, "all-ordered");
        run_zone_group(0, "all-hash");
        if (N >= 2) { run_half_half("half-half"); }
    }

    delete index;
    return 0;
}
