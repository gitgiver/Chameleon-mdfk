//
// Range-query harness for the adaptive-leaf experiment.
//
// The leaf layout is the independent variable (LEAF_WINDOW_THRESHOLD), the index structure is
// either pinned (FIXED_CONF, TARGET_LEAF) or left to the policy, and the workload is one point
// on a selectivity sweep.
//
// Workload construction is quantile-based: each query picks a start index in the sorted dataset
// and extends it by `span = s * n` keys, so the result size is the same for every dataset and the
// two leaf forms are compared on equal work. (A key-width construction, where the width is a
// fraction of the key range, would make the result size depend on local density; that is a
// separate, density-biased workload.)
//
// Correctness: the dataset is sorted, so the expected answer is exactly
// dataset[lower_bound(lo) .. upper_bound(hi)). Every query is checked for size, and a sample is
// checked element-wise as a multiset (hash leaves emit in slot order, ordered leaves in key order).
//
// Env: PILOT_DATASET, PILOT_LENGTH, PILOT_RANGE (selectivity s), PILOT_REPEATS, PILOT_EMIT_VOLUME
// Output: one row per run appended to pilot_range.csv
//

#include <iostream>
#include <random>
#include <functional>
#include <iomanip>
#include <chrono>
#include <cstdio>
#include <string>
#include <torch/torch.h>
#include "../include/DEFINE.h"

#define using_small_network
// per-leaf instrumentation (leaf_stats, leaf_id, the distribution prints below)
#define PILOT_DIAGNOSTICS

#include "../index/include/Index.hpp"
#include "../include/DataSet.hpp"
#include "../index/include/experience.hpp"
#include "../index/include/Controller.hpp"

GlobalController controller;

// "%g" plus dot-to-'p' so the value is safe inside a filename
static std::string tag_num(double v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%g", v);
    std::string s(buf);
    for (char &c: s) { if (c == '.') { c = 'p'; } }
    return s;
}

int main() {
    controller.load_in();
    e.seed(12345);// policy Configuration reproducible when FIXED_CONF is not given

    const char *ds_env = std::getenv("PILOT_DATASET");
    std::string dataset_name = ds_env ? ds_env : "uden.data";
    long length = std::getenv("PILOT_LENGTH") ? std::atol(std::getenv("PILOT_LENGTH")) : 20000000L;

    auto dataset = dataset_source::get_dataset<std::pair<KEY_TYPE, VALUE_TYPE>>(
            data_father_path + dataset_name);
    if ((long) dataset.size() > length) { dataset.resize(length); }
    std::sort(dataset.begin(), dataset.end(),
              [](const std::pair<KEY_TYPE, VALUE_TYPE> &a, const std::pair<KEY_TYPE, VALUE_TYPE> &b) {
                  return a.first < b.first;
              });
    const long n = (long) dataset.size();
    auto min_max = get_min_max<KEY_TYPE, VALUE_TYPE>(dataset.begin(), dataset.end());

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

    const double s = std::getenv("PILOT_RANGE") ? std::atof(std::getenv("PILOT_RANGE")) : 1e-3;

    std::string tag = dataset_name + "_n" + std::to_string(n);
    if (leaf_window_threshold > 0) {
        tag += "_w" + tag_num(leaf_window_threshold);
    } else if (adaptive_theta > 0) {
        tag += "_t" + tag_num(adaptive_theta);
    } else {
        tag += "_base";
    }
    tag += fixed_conf ? ("_conf" + tag_num(fixed_root) + "x" + tag_num(fixed_inner)) : "_rl";
    if (target_leaf > 0) { tag += "_tl" + std::to_string(target_leaf) + "x" + std::to_string(target_fanout); }
    tag += "_r" + tag_num(s);
    printf("run: %s (selectivity=%g fixed_conf=%d target_leaf=%d target_fanout=%d)\n",
           tag.c_str(), s, (int) fixed_conf, target_leaf, target_fanout);

    Hits::node_memory_count = 0;
    Hits::leaf_count = 0;
    Hits::ordered_leaf_count = 0;
    Hits::leaf_skews.clear();
    Hits::leaf_windows.clear();
    Hits::leaf_stats.clear();

    auto index = new Hits::Index<KEY_TYPE, VALUE_TYPE>(conf, min_max.first, min_max.second);
    index->bulk_load(dataset.begin(), dataset.end());

    const double mem_bytes = index->memory_occupied();
    const double mem_ratio = mem_bytes / (double(n) * sizeof(std::pair<KEY_TYPE, VALUE_TYPE>));

    long long leaves_small = 0;
    long long leaves_eligible = 0;
    long long ordered_eligible = 0;
    for (auto &st: Hits::leaf_stats) {
        if (st.size <= DATA_NODE_SIZE) { ++leaves_small; } else {
            ++leaves_eligible;
            ordered_eligible += st.ordered;
        }
    }
    const double ordered_pct_all = 100.0 * double(Hits::ordered_leaf_count)
                                   / double(std::max<unsigned long long>(1, Hits::leaf_count));
    const double ordered_pct_elig = 100.0 * double(ordered_eligible)
                                    / double(std::max<long long>(1, leaves_eligible));

    // Leaf-size and prediction-error distributions. A hash leaf must scan its whole capacity to
    // answer a range query, so the expected cost of a random query is governed by the size of the
    // leaf a random rank lands in -- the size-weighted mean below, not the plain mean.
    if (!Hits::leaf_stats.empty()) {
        std::vector<int> sizes;
        sizes.reserve(Hits::leaf_stats.size());
        long long small = 0;
        double sum = 0;
        double sum_sq = 0;
        for (auto &st: Hits::leaf_stats) {
            sizes.push_back(st.size);
            if (st.size <= DATA_NODE_SIZE) { ++small; }
            sum += double(st.size);
            sum_sq += double(st.size) * double(st.size);
        }
        std::sort(sizes.begin(), sizes.end());
        auto pct = [&](double p) { return sizes[std::min(sizes.size() - 1, (size_t) (p * sizes.size()))]; };
        printf("  leaf size: min=%d p25=%d p50=%d p75=%d p99=%d max=%d | mean=%.1f size_weighted_mean=%.1f | <=%d keys: %.1f%%\n",
               sizes.front(), pct(0.25), pct(0.50), pct(0.75), pct(0.99), sizes.back(),
               sum / double(sizes.size()), sum_sq / std::max(1.0, sum),
               (int) DATA_NODE_SIZE, 100.0 * double(small) / double(sizes.size()));
    }
    if (!Hits::leaf_windows.empty()) {
        auto windows = Hits::leaf_windows;
        std::sort(windows.begin(), windows.end());
        auto wq = [&](double p) { return windows[std::min(windows.size() - 1, (size_t) (p * windows.size()))]; };
        printf("  window(p99,slots): min=%.2f p25=%.2f p50=%.2f p90=%.2f p99=%.2f max=%.2f\n",
               windows.front(), wq(0.25), wq(0.50), wq(0.90), wq(0.99), windows.back());
    }
    fflush(stdout);

    // ---- workload: quantile construction, query set fixed across passes ----
    const long span = std::max(1L, (long) (s * double(n)));
    const long emit_budget = std::getenv("PILOT_EMIT_VOLUME") ? std::atol(std::getenv("PILOT_EMIT_VOLUME")) : 50000000L;
    const long queries = std::max(200L, std::min(2000000L, emit_budget / std::max(1L, span)));

    std::mt19937_64 rng(1000);
    std::uniform_int_distribution<long> dist(0, std::max<long>(0, n - span - 1));
    std::vector<long> starts(queries);
    for (long q = 0; q < queries; ++q) { starts[q] = dist(rng); }

    const char *rep_env = std::getenv("PILOT_REPEATS");
    const int repeats = rep_env ? std::max(1, std::atoi(rep_env)) : 5;

    std::vector<std::pair<KEY_TYPE, VALUE_TYPE>> out;
    out.reserve((size_t) std::min<long>(2 * span + 16, 2000000L));
    long long emitted = 0;
    std::vector<double> pass_ns;
    pass_ns.reserve(repeats);
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
    std::vector<double> measured(pass_ns.begin() + (pass_ns.size() > 1 ? 1 : 0), pass_ns.end());
    std::sort(measured.begin(), measured.end());
    const double ns = measured.front();
    const double ns_median = measured[measured.size() / 2];
    const double ns_max = measured.back();
    const double ns_per_rep = double(emitted) / double(queries * repeats);

    // ---- correctness: size for every query, multiset equality for a sample (untimed) ----
    long errors = 0;
    long verified = 0;
    const long verify_deep = std::min<long>(queries, 200);
    auto by_key = [](const std::pair<KEY_TYPE, VALUE_TYPE> &a, KEY_TYPE v) { return a.first < v; };
    auto above_key = [](KEY_TYPE v, const std::pair<KEY_TYPE, VALUE_TYPE> &a) { return v < a.first; };
    for (long q = 0; q < queries; ++q) {
        const long st = starts[q];
        const KEY_TYPE lo = dataset[st].first;
        const KEY_TYPE hi = dataset[st + span].first;
        index->range_query(lo, hi, out);
        auto a = std::lower_bound(dataset.begin(), dataset.end(), lo, by_key);
        auto b = std::upper_bound(dataset.begin(), dataset.end(), hi, above_key);
        if ((long) out.size() != (long) (b - a)) {
            ++errors;
            continue;
        }
        if (q < verify_deep) {
            std::sort(out.begin(), out.end(),
                      [](const std::pair<KEY_TYPE, VALUE_TYPE> &x, const std::pair<KEY_TYPE, VALUE_TYPE> &y) {
                          return x.first < y.first;
                      });
            if (!std::equal(out.begin(), out.end(), a)) { ++errors; }
            ++verified;
        }
    }

    printf("selectivity=%-8g span=%-8ld queries=%-8ld leaves=%-6llu ordered=%.1f%%(elig %.1f%%) "
           "mem_ratio=%-7.4f result_avg=%-9.1f ns=%-9.2f (med %.2f max %.2f, %d passes) "
           "ns_per_result=%.4f errors=%ld\n",
           s, span, queries, Hits::leaf_count, ordered_pct_all, ordered_pct_elig,
           mem_ratio, ns_per_rep, ns, ns_median, ns_max, (int) measured.size(),
           ns / std::max(1.0, ns_per_rep), errors);
    fflush(stdout);

    {
        const char *results = "pilot_range.csv";
        FILE *probe = std::fopen(results, "r");
        bool need_header = (probe == nullptr);
        if (probe) { std::fclose(probe); }
        FILE *f = std::fopen(results, "a");
        if (f) {
            if (need_header) {
                std::fprintf(f, "tag,dataset,n,selectivity,span,queries,conf_source,root_fanout,target_leaf,"
                                "target_fanout,leaves,leaves_small,leaves_eligible,ordered,ordered_pct_all,"
                                "ordered_pct_eligible,mem_ratio,result_avg,ns_min,ns_median,ns_max,"
                                "ns_per_result,passes,verified_queries,errors\n");
            }
            std::fprintf(f, "%s,%s,%ld,%.6g,%ld,%ld,%s,%.0f,%d,%d,%llu,%lld,%lld,%llu,%.3f,%.3f,%.6f,"
                            "%.2f,%.3f,%.3f,%.3f,%.5f,%d,%ld,%ld\n",
                         tag.c_str(), dataset_name.c_str(), n, s, span, queries,
                         fixed_conf ? "fixed" : "policy", conf.root_fan_out, target_leaf, target_fanout,
                         Hits::leaf_count, leaves_small, leaves_eligible, Hits::ordered_leaf_count,
                         ordered_pct_all, ordered_pct_elig, mem_ratio, ns_per_rep,
                         ns, ns_median, ns_max, ns / std::max(1.0, ns_per_rep),
                         (int) measured.size(), verified, errors);
            std::fclose(f);
        }
    }
    fflush(stdout);

    delete index;
    return 0;
}
