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
// Env: PROD_DATASET (default uden.data), PROD_LENGTH, PROD_QUERIES, PROD_RANGE, PROD_REPEATS,
//      FIXED_CONF=<root>,<inner> overrides the agents' structure choice
//

#include <iostream>
#include <random>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
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

    delete index;
    return 0;
}
