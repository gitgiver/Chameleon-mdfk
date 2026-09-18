//
// Pilot harness for the adaptive-leaf experiment.
//
// Structure. By default both agents are in play: GlobalController (DARE, AC_Q_Net.pt) picks the
// Configuration and Small_Q_network (TSMDP, tmp.pt) picks the per-region fanout during build_.
// A comparison is only attributable to the leaf layout if the tree shape and the leaf sizes are
// held fixed, so two deterministic overrides exist:
//   FIXED_CONF=<root_fanout>,<inner_fanout> -> bypass DARE, use this Configuration everywhere
//   TARGET_LEAF=<n> [TARGET_FANOUT=<b>]     -> bypass TSMDP, split until a node holds <= n keys
// With both set, every dataset is built on the same shape with the same leaf sizes, and the leaf
// layout becomes the only independent variable.
//
// Leaf layout criterion (the independent variable):
//   LEAF_WINDOW_THRESHOLD > 0 -> ordered leaf iff the leaf's p99 rank-prediction error, in slots,
//                                is <= threshold; takes precedence over ADAPTIVE_THETA
//   ADAPTIVE_THETA > 0        -> ordered leaf iff local skew <= theta
//   neither                   -> all hash leaves (baseline)
//
// Env: PILOT_DATASET (default uden.data), PILOT_LENGTH, PILOT_QUERIES. Seeds are fixed
// (GA 12345, query stream 1000) so runs are deterministic and comparable.
// Output: one row per run appended to pilot_results.csv, plus leaf_stats_<tag>.csv.
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
// per-leaf instrumentation (leaf_stats, leaf_id, the distribution prints below). Costs 8 bytes
// per leaf and writes shared state on every lookup, so it is off unless a harness asks for it.
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
    // make the GA-driven Configuration reproducible so theta is the only variable
    e.seed(12345);

    const char *ds_env = std::getenv("PILOT_DATASET");
    std::string dataset_name = ds_env ? ds_env : "uden.data";
    long length = std::getenv("PILOT_LENGTH") ? std::atol(std::getenv("PILOT_LENGTH")) : 20000000L;
    long n_queries = std::getenv("PILOT_QUERIES") ? std::atol(std::getenv("PILOT_QUERIES")) : 2000000L;

    auto dataset = dataset_source::get_dataset<std::pair<KEY_TYPE, VALUE_TYPE>>(
            data_father_path + dataset_name);
    if ((long) dataset.size() > length) { dataset.resize(length); }
    std::sort(dataset.begin(), dataset.end(),
              [](std::pair<KEY_TYPE, VALUE_TYPE> &a, std::pair<KEY_TYPE, VALUE_TYPE> &b) {
                  return a.first < b.first;
              });

    long n = (long) dataset.size();
    auto min_max = get_min_max<KEY_TYPE, VALUE_TYPE>(dataset.begin(), dataset.end());
    auto pdf = get_pdf<KEY_TYPE, VALUE_TYPE>(dataset.begin(), dataset.end(),
                                             min_max.first, min_max.second, BUCKET_SIZE);

    experience_t exp_chosen;
    std::copy(pdf.begin(), pdf.end(), exp_chosen.distribution);
    exp_chosen.data_size = float(n);

    // structure: policy by default, deterministic override for controlled comparisons
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
        conf = controller.get_best_action_GA(exp_chosen).conf;
    }

    // run tag: identifies the CSV and the row appended to pilot_results.csv
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
    printf("run: %s (fixed_conf=%d target_leaf=%d target_fanout=%d)\n",
           tag.c_str(), (int) fixed_conf, target_leaf, target_fanout);

    Hits::node_memory_count = 0;
    Hits::leaf_count = 0;
    Hits::ordered_leaf_count = 0;
    Hits::inner_cost = 0;
    Hits::leaf_cost = 0;
    Hits::leaf_max_cost = 0;
    Hits::leaf_skews.clear();
    Hits::leaf_windows.clear();
    Hits::ordered_probe_sum = 0;
    Hits::ordered_lookup_count = 0;
    Hits::leaf_stats.clear();

    auto index = new Hits::Index<KEY_TYPE, VALUE_TYPE>(conf, min_max.first, min_max.second);
    index->bulk_load(dataset.begin(), dataset.end());

    double mem_bytes = index->memory_occupied();
    double mem_ratio = mem_bytes / (double(n) * sizeof(std::pair<KEY_TYPE, VALUE_TYPE>));

    // Latency measurement. A single pass is not reproducible: four runs of an identical
    // configuration (uden, where every threshold yields 100% ordered leaves and the same
    // memory) spread over 414-638 ns/query. Repeat the query stream, discard the first pass
    // as warm-up, and report the distribution of per-pass means. The minimum is the robust
    // point estimate, since interference can only make a pass slower.
    const char *rep_env = std::getenv("PILOT_REPEATS");
    const int repeats = rep_env ? std::max(1, std::atoi(rep_env)) : 5;
    const long per_pass = std::max<long>(1, n_queries / repeats);

    std::mt19937_64 rng(1000);
    std::uniform_int_distribution<long> dist(0, n - 1);
    VALUE_TYPE value;
    long errors = 0;
    std::vector<double> pass_ns;
    pass_ns.reserve(repeats);
    for (int rep = 0; rep < repeats; ++rep) {
        Hits::inner_cost = 0;
        Hits::leaf_cost = 0;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (long q = 0; q < per_pass; ++q) {
            long id = dist(rng);
            if (!index->get_with_cost(dataset[id].first, value) || value != dataset[id].second) {
                ++errors;
            }
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        pass_ns.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count() / double(per_pass));
    }
    std::vector<double> measured(pass_ns.begin() + (pass_ns.size() > 1 ? 1 : 0), pass_ns.end());
    std::sort(measured.begin(), measured.end());
    const double ns = measured.front();                        // minimum = least perturbed pass
    const double ns_median = measured[measured.size() / 2];
    const double ns_max = measured.back();
    const long n_measured = per_pass * (long) measured.size();

    // Rate denominators. DATA_NODE_SIZE is both the recursion floor and the ordered-leaf
    // eligibility floor, so leaves at or below it can never be ordered and dilute the headline
    // percentage. Report against the eligible population as well.
    long long leaves_small = 0;
    long long leaves_eligible = 0;
    long long ordered_eligible = 0;
    for (auto &s: Hits::leaf_stats) {
        if (s.size <= DATA_NODE_SIZE) {
            ++leaves_small;
        } else {
            ++leaves_eligible;
            ordered_eligible += s.ordered;
        }
    }
    double ordered_pct_all = 100.0 * double(Hits::ordered_leaf_count)
                             / double(std::max<unsigned long long>(1, Hits::leaf_count));
    double ordered_pct_elig = 100.0 * double(ordered_eligible) / double(std::max<long long>(1, leaves_eligible));

    printf("theta=%-10g dataset=%-12s n=%-10ld root=%-8.0f mem_MB=%-9.2f mem_ratio=%-7.4f "
           "get_ns=%-8.2f (med %.2f max %.2f, %d passes x %ld) "
           "leaves=%-6llu ordered=%-6llu(%.1f%%) inner=%-10lld leaf=%-10lld errors=%ld\n",
           adaptive_theta, dataset_name.c_str(), n, conf.root_fan_out,
           mem_bytes / (1024.0 * 1024.0), mem_ratio, ns, ns_median, ns_max, (int) measured.size(),
           per_pass, Hits::leaf_count, Hits::ordered_leaf_count, ordered_pct_all,
           Hits::inner_cost, Hits::leaf_cost, errors);
    printf("  ordered rate: %.1f%% of all leaves, %.1f%% of the %lld leaves with >%d keys "
           "(%lld leaves have <=%d keys, %.1f%%)\n",
           ordered_pct_all, ordered_pct_elig, leaves_eligible, (int) DATA_NODE_SIZE,
           leaves_small, (int) DATA_NODE_SIZE,
           100.0 * double(leaves_small) / double(std::max<long long>(1, leaves_small + leaves_eligible)));
    if (Hits::ordered_lookup_count > 0) {
        printf("  ordered-leaf prediction: avg searched window = %.2f slots (over %lld lookups)\n",
               double(Hits::ordered_probe_sum) / double(Hits::ordered_lookup_count),
               Hits::ordered_lookup_count);
    }
    printf("  criterion: window_threshold=%.6g  theta=%.6g\n",
           leaf_window_threshold, adaptive_theta);
    fflush(stdout);

    // per-leaf p99 prediction error: the statistic the window criterion thresholded
    if (!Hits::leaf_windows.empty()) {
        auto w = Hits::leaf_windows;
        std::sort(w.begin(), w.end());
        auto pct = [&](double p) { return w[std::min(w.size() - 1, (size_t) (p * w.size()))]; };
        printf("  window(p99,slots): min=%.2f p25=%.2f p50=%.2f p75=%.2f p99=%.2f max=%.2f  (leaves=%zu)\n",
               w.front(), pct(0.25), pct(0.50), pct(0.75), pct(0.99), w.back(), w.size());
        fflush(stdout);
    }

    // leaf size distribution: DATA_NODE_SIZE doubles as the recursion floor and as the
    // ordered-leaf eligibility floor, so tiny leaves pile up in sparse regions
    if (!Hits::leaf_stats.empty()) {
        std::vector<int> sz;
        sz.reserve(Hits::leaf_stats.size());
        for (auto &s: Hits::leaf_stats) { sz.push_back(s.size); }
        std::sort(sz.begin(), sz.end());
        auto pct = [&](double p) { return sz[std::min(sz.size() - 1, (size_t) (p * sz.size()))]; };
        long long tiny = 0;
        for (int v: sz) { if (v <= DATA_NODE_SIZE) { ++tiny; } }
        printf("  leaf size: min=%d p25=%d p50=%d p75=%d p99=%d max=%d  |  <=%d keys: %lld (%.1f%%)\n",
               sz.front(), pct(0.25), pct(0.50), pct(0.75), pct(0.99), sz.back(),
               (int) DATA_NODE_SIZE, tiny, 100.0 * double(tiny) / double(sz.size()));
        fflush(stdout);
    }

    // per-leaf skew distribution
    if (!Hits::leaf_skews.empty()) {
        auto s = Hits::leaf_skews;
        std::sort(s.begin(), s.end());
        double mn = s.front(), mx = s.back();
        auto pct = [&](double p) { return s[std::min(s.size() - 1, (size_t) (p * s.size()))]; };
        printf("  skew: min=%.4f p25=%.4f p50=%.4f p75=%.4f max=%.4f  (leaves=%zu)\n",
               mn, pct(0.25), pct(0.50), pct(0.75), mx, s.size());
        const int B = 20;
        std::vector<long long> hist(B, 0);
        double w = (mx - mn) / B;
        if (w <= 0) { w = 1; }
        for (double v: s) {
            int b = std::min(B - 1, std::max(0, (int) ((v - mn) / w)));
            ++hist[b];
        }
        long long hmax = *std::max_element(hist.begin(), hist.end());
        for (int b = 0; b < B; ++b) {
            int bar = (int) (60.0 * double(hist[b]) / double(std::max(1LL, hmax)));
            printf("   [%8.4f] %9lld %s\n", mn + b * w, hist[b], std::string(std::max(0, bar), '#').c_str());
        }
    }
    fflush(stdout);

    // machine-readable row for the sweep table; header written once
    {
        const char *results = "pilot_results.csv";
        FILE *probe = std::fopen(results, "r");
        bool need_header = (probe == nullptr);
        if (probe) { std::fclose(probe); }
        FILE *f = std::fopen(results, "a");
        if (f) {
            if (need_header) {
                std::fprintf(f, "tag,dataset,n,window_threshold,theta,conf_source,root_fanout,fixed_inner,"
                                "target_leaf,target_fanout,leaves,leaves_small,leaves_eligible,ordered,"
                                "ordered_pct_all,ordered_pct_eligible,mem_ratio,get_ns_min,get_ns_median,"
                                "get_ns_max,passes,queries_per_pass,errors\n");
            }
            std::fprintf(f, "%s,%s,%ld,%.6g,%.6g,%s,%.0f,%.0f,%d,%d,%llu,%lld,%lld,%llu,%.3f,%.3f,%.6f,%.3f,%.3f,%.3f,%d,%ld,%ld\n",
                         tag.c_str(), dataset_name.c_str(), n,
                         leaf_window_threshold, adaptive_theta,
                         fixed_conf ? "fixed" : "policy", conf.root_fan_out,
                         fixed_conf ? fixed_inner : -1.0,
                         target_leaf, target_fanout,
                         Hits::leaf_count, leaves_small, leaves_eligible, Hits::ordered_leaf_count,
                         ordered_pct_all, ordered_pct_elig, mem_ratio, ns, ns_median, ns_max,
                         (int) measured.size(), per_pass, errors);
            std::fclose(f);
        }
    }

    // per-leaf diagnostics: size, layout, skew, lookups, avg search steps -> lets us tell whether
    // the ordered-leaf cost is driven by leaf size, by the key layout, or by local skew.
    {
        std::string csv = std::string("leaf_stats_") + tag + ".csv";
        FILE *f = std::fopen(csv.c_str(), "w");
        if (f) {
            std::fprintf(f, "leaf_id,size,ordered,skew,window_p99,window_max,interval_width,key_span,"
                            "lower,upper,lookups,avg_probe\n");
            for (std::size_t i = 0; i < Hits::leaf_stats.size(); ++i) {
                Hits::LeafStat &s = Hits::leaf_stats[i];
                if (s.lookups > 0) {
                    std::fprintf(f, "%zu,%d,%d,%.6f,%.4f,%.4f,%.6g,%.6g,%.10g,%.10g,%lld,%.4f\n",
                                 i, s.size, s.ordered, s.skew, s.window_p99, s.window_max,
                                 s.interval_width, s.key_span, s.lower, s.upper,
                                 s.lookups, double(s.probe) / double(s.lookups));
                }
            }
            std::fclose(f);
            printf("  wrote %s\n", csv.c_str());
        }
    }
    fflush(stdout);

    delete index;
    return 0;
}
