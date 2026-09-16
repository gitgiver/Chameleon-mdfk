//
// Pilot harness for the adaptive-leaf experiment.
// Runs the REAL Chameleon index with BOTH agents in play:
//   - GlobalController (DARE, AC_Q_Net.pt) picks the Configuration (fanout matrix)
//   - Small_Q_network (TSMDP, tmp.pt) picks the per-region fanout during build_
// The leaf layout is the only knob changed, via one of two criteria:
//   ADAPTIVE_THETA > 0        -> regions with local skew <= theta become ordered dense leaves
//   LEAF_WINDOW_THRESHOLD > 0 -> regions whose p99 prediction error (slots) <= threshold become
//                                ordered dense leaves; takes precedence over ADAPTIVE_THETA
//   neither                   -> all hash leaves (baseline)
//
// Env: PILOT_DATASET (default uden.data), PILOT_LENGTH, PILOT_QUERIES
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

#include "../index/include/Index.hpp"
#include "../include/DataSet.hpp"
#include "../index/include/experience.hpp"
#include "../index/include/Controller.hpp"

GlobalController controller;

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
    auto conf = controller.get_best_action_GA(exp_chosen).conf;

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

    std::mt19937_64 rng(1000);
    std::uniform_int_distribution<long> dist(0, n - 1);
    VALUE_TYPE value;
    Hits::inner_cost = 0;
    Hits::leaf_cost = 0;
    long errors = 0;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (long q = 0; q < n_queries; ++q) {
        long id = dist(rng);
        if (!index->get_with_cost(dataset[id].first, value) || value != dataset[id].second) {
            ++errors;
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / double(n_queries);

    printf("theta=%-10g dataset=%-12s n=%-10ld root=%-8.0f mem_MB=%-9.2f mem_ratio=%-7.4f get_ns=%-8.2f "
           "leaves=%-6llu ordered=%-6llu(%.1f%%) inner=%-10lld leaf=%-10lld errors=%ld\n",
           adaptive_theta, dataset_name.c_str(), n, conf.root_fan_out,
           mem_bytes / (1024.0 * 1024.0), mem_ratio, ns,
           Hits::leaf_count, Hits::ordered_leaf_count,
           100.0 * double(Hits::ordered_leaf_count) / double(std::max<unsigned long long>(1, Hits::leaf_count)),
           Hits::inner_cost, Hits::leaf_cost, errors);
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

    // per-leaf diagnostics: size, skew, lookups, avg search steps -> lets us tell
    // whether the ordered-leaf cost is driven by leaf size or by local skew.
    {
        std::string csv = std::string("leaf_stats_") + dataset_name + ".csv";
        FILE *f = std::fopen(csv.c_str(), "w");
        if (f) {
            std::fprintf(f, "leaf_id,size,skew,window_p99,window_max,interval_width,key_span,lookups,avg_probe\n");
            for (std::size_t i = 0; i < Hits::leaf_stats.size(); ++i) {
                Hits::LeafStat &s = Hits::leaf_stats[i];
                if (s.lookups > 0) {
                    std::fprintf(f, "%zu,%d,%.6f,%.4f,%.4f,%.6g,%.6g,%lld,%.4f\n", i, s.size, s.skew,
                                 s.window_p99, s.window_max, s.interval_width, s.key_span, s.lookups,
                                 double(s.probe) / double(s.lookups));
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
