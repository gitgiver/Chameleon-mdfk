//
// Created by redamancyguy on 23-5-30.
//

#ifndef INDEX_H
#define INDEX_H

#include <thread>
#include <stack>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include "Configuration.hpp"
#include "RL_network.hpp"

#define DATA_NODE_SIZE 5
const float min_density = 0.35;
// DEFAULT_DENSITY overrides the leaf hash load factor (size/capacity). A measurement hook for the
// per-leaf memory-vs-probe sweep; unset -> the original constant.
static float default_density = []() -> float {
    const char *s = std::getenv("DEFAULT_DENSITY");
    return s ? std::atof(s) : 0.65f;
}();
const float max_density = 0.95;


//const float min_density = 0.5;
//const float default_density = 0.7;
//const float max_density = 0.9;

//float calculate_density(float size, float probability = 0.9){
//    return size * (size - 1) / -(std::log(1-probability));
//}
double hash_factor = 10231023.353;
//double hash_factor=1;
//double  hash_factor = 131.3;
// >>> ADAPTIVE LEAF PILOT <<<
// theta <= 0 : disabled  -> every leaf is the original hash leaf (baseline behaviour)
// theta >  0 : a leaf region whose local skew <= theta becomes an ordered, densely
//              packed (density 1.0) key-sorted leaf; otherwise the hash leaf is kept.
// Set through the ADAPTIVE_THETA environment variable so a single binary can sweep it.
static double adaptive_theta = []() -> double {
    const char *s = std::getenv("ADAPTIVE_THETA");
    return s ? std::atof(s) : -1.0;
}();
// Leaf-form criterion #2: threshold on the leaf's own measured prediction error, in slots
// (the "window" a point lookup has to search). Unlike a skew threshold this carries units,
// so one value applies across leaf sizes. LEAF_WINDOW_THRESHOLD > 0 enables it and takes
// precedence over ADAPTIVE_THETA.
static double leaf_window_threshold = []() -> double {
    const char *s = std::getenv("LEAF_WINDOW_THRESHOLD");
    return s ? std::atof(s) : -1.0;
}();
// >>> FIXED-STRUCTURE EXPERIMENT MODE <<<
// Deterministic replacement for the TSMDP fanout decision so that the tree shape and the leaf
// sizes are set by the experiment rather than chosen per dataset by the policy. Needed to
// attribute a measured difference to the leaf layout: with the policy in the loop, uden gets
// ~142-key leaves and face ~14.7k-key leaves, so "ordered is cheap on uniform data" is
// confounded with "its leaves are smaller". Both default to off, so the production path and
// every existing measurement are unchanged.
//   TARGET_LEAF   > 0 : keep splitting until a node holds at most this many keys
//   TARGET_FANOUT     : branching factor for those splits (default 16)
static int target_leaf = []() -> int {
    const char *s = std::getenv("TARGET_LEAF");
    return s ? std::atoi(s) : -1;
}();
static int target_fanout = []() -> int {
    const char *s = std::getenv("TARGET_FANOUT");
    return (s && std::atoi(s) > 0) ? std::atoi(s) : 16;
}();
// LEAF_COST_DECISION > 0 : decide ordered vs hash by measuring both leaf forms' (query + memory)
// cost under the agents' reward weights, instead of a window/skew threshold. See build_leaf_.
static int leaf_cost_decision = []() -> int {
    const char *s = std::getenv("LEAF_COST_DECISION");
    return s ? std::atoi(s) : 0;
}();
// LEAF_RANGE_WEIGHT : weight of a range query relative to a point query, used only by the
// ordered-vs-hash leaf decision below. 0 drops the range term (point + memory only, the default).
// It is deliberately separate from QUERY_WEIGHT/MEMORY_WEIGHT, which the agents' reward also uses.
static double leaf_range_weight = []() -> double {
    const char *s = std::getenv("LEAF_RANGE_WEIGHT");
    return s ? std::atof(s) : 0.0;
}();
// >>> WORKLOAD-AWARE LEAF DENSITY <<<
// The index is built for a workload, not only for the data: the harness fills workload_model with
// a key-space histogram of the query distribution BEFORE bulk_load, and each hash leaf's load
// factor is set from that leaf's own query rate. A hot leaf (queried at or above the global
// average per-key rate) keeps default_density; a cold leaf is packed to WORKLOAD_COLD_DENSITY.
// Empty model -> every leaf keeps default_density, i.e. the baseline path, byte for byte.
//   WORKLOAD_DENSITY > 0 : apply the decision
//   unset                : only accumulate kappa/phi (the "count-only" measurement mode)
struct WorkloadModel {
    double lower = 0;   // key bounds of the histogram (the root interval)
    double upper = 0;
    int bins = 0;
    double keys = 0;    // total keys in the dataset
    double total = 0;   // total queries represented by hist
    std::vector<double> hist;
    bool active() const { return bins > 0 && !hist.empty(); }
};
static WorkloadModel workload_model;
struct WorkloadStats {
    long long leaves = 0, cold_leaves = 0;
    long long all_keys = 0, cold_keys = 0;
    double all_q = 0, cold_q = 0;
};
static WorkloadStats workload_stats;
static double workload_cold_density = []() -> double {
    const char *s = std::getenv("WORKLOAD_COLD_DENSITY");
    return s ? std::atof(s) : 0.85;
}();
static int workload_apply = []() -> int {
    const char *s = std::getenv("WORKLOAD_DENSITY");
    return s ? std::atoi(s) : 0;
}();
// RUNTIME_ADAPTIVE > 0 : hash leaves count their accesses and the index can rebalance leaf densities
// from live traffic (no build-time workload model needed). Off by default, so the build-time path is
// unchanged. The counters are only read/written by the harness via Index::rebalance_leaves().
static int runtime_adaptive = []() -> int {
    const char *s = std::getenv("RUNTIME_ADAPTIVE");
    return s ? std::atoi(s) : 0;
}();

// Query mass in the histogram over a leaf's key interval [lower, upper].
static double workload_leaf_q(double lower, double upper) {
    if (!workload_model.active()) { return 0.0; }
    const double span = workload_model.upper - workload_model.lower;
    if (!(span > 0)) { return 0.0; }
    auto bin_of = [&](double key) {
        double p = double(workload_model.bins) * (key - workload_model.lower) / span;
        int b = int(p);
        if (b < 0) { b = 0; }
        if (b >= workload_model.bins) { b = workload_model.bins - 1; }
        return b;
    };
    double q = 0;
    for (int b = bin_of(lower); b <= bin_of(upper); ++b) { q += workload_model.hist[b]; }
    return q;
}

// True when a leaf with this key interval and size sits below the global mean per-key query rate.
// build_leaf_ uses it to call a leaf cold; the harness uses it to label queries the same way, so a
// query's class always matches the class the layout was built with.
static bool workload_leaf_is_cold(int data_count, double lower, double upper) {
    if (!workload_model.active() || workload_model.keys <= 0) { return false; }
    const double avg = workload_model.total / workload_model.keys;
    const double rate = (data_count > 0) ? workload_leaf_q(lower, upper) / double(data_count) : 0.0;
    return rate < avg;
}

// Load factor to build a leaf at. Side effect: accumulates the kappa (cold key share) / phi (cold
// query share) statistics that decide whether a workload-aware layout is worth anything at all.
static double workload_leaf_density(int data_count, double lower, double upper) {
    if (!workload_model.active()) { return default_density; }
    const double q = workload_leaf_q(lower, upper);
    const bool cold = workload_leaf_is_cold(data_count, lower, upper);
    ++workload_stats.leaves;
    workload_stats.all_keys += data_count;
    workload_stats.all_q += q;
    if (cold) {
        ++workload_stats.cold_leaves;
        workload_stats.cold_keys += data_count;
        workload_stats.cold_q += q;
    }
    if (!workload_apply) { return default_density; }
    return cold ? workload_cold_density : default_density;
}
//#define inner_cost_weight (float(6.01213))
#define inner_cost_weight (float(60.01213))
#define leaf_cost_weight (float(1.35074178))

inline double forward_with_parameters(double lower, double upper, double capacity, double x) {
    return capacity * ((x - lower) / (upper - lower));
}

inline bool get_bitmap(const unsigned int *bitmap, const unsigned int position) {
    unsigned int index = position >> 5;
    unsigned int bit_index = position - (index << 5);
    return (bitmap[index] >> bit_index) & 1;
}

inline void de_set_bitmap(unsigned int *bitmap, const unsigned int position) {
    unsigned int index = position >> 5;
    unsigned int bit_index = position - (index << 5);
    bitmap[index] &= ~(1 << bit_index);
}

inline void set_bitmap(unsigned int *bitmap, const unsigned int position) {
    unsigned int index = position >> 5;
    unsigned int bit_index = position - (index << 5);
    bitmap[index] |= 1 << bit_index;
}

// Same measure as DataSet.hpp local_skew, inlined here to avoid pulling that header in.
// ~1 for locally uniform regions, >> 1 for locally skewed (spiky) regions.
template<class key_T, class value_T>
inline double leaf_skew(
        typename std::vector<std::pair<key_T, value_T>>::const_iterator begin,
        typename std::vector<std::pair<key_T, value_T>>::const_iterator end,
        double lower, double upper) {
    double n_1 = double(end - begin) - 1;
    if (n_1 <= 0) { return 1.0; }
    double sum = 0;
    for (auto s = begin + 1; s < end; ++s) {
        double gl = ((s - 1)->first - lower) / (upper - lower);
        double gr = (s->first - lower) / (upper - lower);
        double gap = gr - gl;
        if (gap <= 1e-18) { gap = 1e-18; }
        sum += 1.0 / gap;
    }
    return sum / (n_1 * n_1);
}

// Point-lookup cost model for a candidate ordered leaf. The linear model ranks a key at
// p = (n-1)*(key-kmin)/(kmax-kmin); a lookup that misses must search |p - actual_rank|
// slots. Reports the 99th percentile and the maximum of that error over the leaf's keys.
// Requires keys sorted by key. Costs one O(n) pass, no extra sort.
template<class key_T, class value_T>
inline void leaf_prediction_error(
        const std::vector<std::pair<key_T, value_T>> &keys,
        double &p99_window, double &max_window) {
    p99_window = 0;
    max_window = 0;
    const int n = int(keys.size());
    if (n <= 1) { return; }
    const double kmin = double(keys.front().first);
    const double kmax = double(keys.back().first);
    if (!(kmax > kmin)) { return; }//degenerate: all keys identical
    std::vector<int> errors(n);
    for (int i = 0; i < n; ++i) {
        double p = double(n - 1) * (double(keys[i].first) - kmin) / (kmax - kmin);
        if (p < 0) { p = 0; }
        if (p > n - 1) { p = n - 1; }
        errors[i] = std::abs(int(std::llround(p)) - i);
    }
    const std::size_t k = std::size_t(0.99 * double(n - 1));
    std::nth_element(errors.begin(), errors.begin() + k, errors.end());
    p99_window = double(errors[k]);
    max_window = double(*std::max_element(errors.begin(), errors.end()));
}

// Mean point-lookup step count of the ordered (dense sorted) leaf, averaged over its keys: the
// prediction access, then the gallop and binary search of find_ordered. Mirrors find_ordered so the
// number is the work a lookup actually pays. Used by the two-sided leaf decision (build_leaf_).
template<class key_T, class value_T>
inline double mean_ordered_steps(const std::vector<std::pair<key_T, value_T>> &keys) {
    const int n = int(keys.size());
    if (n <= 1) { return 1.0; }
    const double kmin = double(keys.front().first);
    const double kmax = double(keys.back().first);
    if (!(kmax > kmin)) { return 1.0; }//degenerate: all keys identical
    long long total = 0;
    for (int i = 0; i < n; ++i) {
        const double key = double(keys[i].first);
        int p = int(double(n - 1) * (key - kmin) / (kmax - kmin));
        if (p < 0) { p = 0; }
        if (p > n - 1) { p = n - 1; }
        long long steps = 1;//the prediction access
        if (double(keys[p].first) == key) { total += steps; continue; }
        int lo;
        int hi;
        if (double(keys[p].first) < key) {//key lies to the right of the prediction
            int last = p;
            int d = 1;
            while (p + d < n && double(keys[p + d].first) < key) { last = p + d; d <<= 1; ++steps; }
            lo = last + 1;
            hi = std::min(n - 1, p + d);
        } else {//key lies to the left of the prediction
            int last = p;
            int d = 1;
            while (p - d >= 0 && double(keys[p - d].first) > key) { last = p - d; d <<= 1; ++steps; }
            lo = std::max(0, p - d);
            hi = last - 1;
        }
        while (lo <= hi) {
            ++steps;
            int mid = (lo + hi) >> 1;
            if (double(keys[mid].first) == key) { break; }
            if (double(keys[mid].first) < key) { lo = mid + 1; } else { hi = mid - 1; }
        }
        total += steps;
    }
    return double(total) / double(n);
}

namespace Hits {
    unsigned long long node_memory_count = 0;
    long long inner_cost = 0;
    long long leaf_cost = 0;
    long long leaf_max_cost = 0;

#ifdef PILOT_DIAGNOSTICS
    // Per-leaf instrumentation used by the pilot harnesses. Off by default: it costs 8 bytes per
    // leaf node and, worse, every lookup on an ordered leaf writes these, which is a data race
    // once more than one thread reads concurrently. Build with -DPILOT_DIAGNOSTICS to measure.
    static constexpr bool pilot_diagnostics = true;
    unsigned long long leaf_count = 0;
    unsigned long long ordered_leaf_count = 0;
    std::vector<double> leaf_skews;//per-leaf local skew, for diagnosing the threshold
    std::vector<double> leaf_windows;//per-leaf p99 prediction error, in slots
    long long ordered_probe_sum = 0;//sum of searched window sizes over ordered-leaf lookups
    long long ordered_lookup_count = 0;

    // per-leaf diagnostics: lets us tell whether ordered-leaf cost is driven by
    // the leaf's size or by its local skew.
    struct LeafStat {
        int size;
        int ordered;//1 if this leaf was laid out ordered; lets rates be recomputed on subsets
        double skew;
        long long lookups;
        long long probe;
        double window_p99;
        double window_max;
        double interval_width;//upper - lower: the span leaf_skew normalizes by
        double key_span;//max key - min key actually present, i.e. what a linear model spans
        double lower;//interval bounds, so the same leaf can be identified across runs
        double upper;
    };
    std::vector<LeafStat> leaf_stats;
#else
    static constexpr bool pilot_diagnostics = false;
#endif

    template<class key_T, class value_T>
    class DataNode {
    public:
        using self_type = DataNode;
        using slot_type = std::pair<key_T, value_T>;
    public:
        double lower{0};
        double upper{0};
        int capacity{-1};

        [[nodiscard]] inline double forward(double key) const {
            return std::min(double(capacity - 1), std::max(0.0, forward_with_parameters(lower, upper, capacity, key)));
        }

        ///////
        int size{};
        int max_offset{};
        int ordered{};//1 -> dense key-sorted leaf (pilot); 0 -> original hash leaf
        unsigned int hits{};//runtime adaptation: accesses to this leaf since the last check
#ifdef PILOT_DIAGNOSTICS
        int leaf_id{-1};//index into Hits::leaf_stats, for diagnostics
#endif
        slot_type array[0];


        inline unsigned int *bitmap_start() {
            return (unsigned int *) (array + capacity);
        }

        inline
        static self_type *
        new_segment(int capacity, double lower, double upper) {
            int t = ((capacity >> 5) + 1);
            auto memory_size = sizeof(self_type) + sizeof(slot_type) * capacity + sizeof(unsigned int) * t;
            auto node = (self_type *) std::malloc(memory_size);
            node->capacity = capacity;
            node->lower = lower;
            node->upper = upper;
            node->size = 0;
            node->max_offset = 0;
            node->ordered = 0;
            node->hits = 0;
#ifdef PILOT_DIAGNOSTICS
            node->leaf_id = -1;
#endif
            unsigned int *bit_map = node->bitmap_start();
            for (int i = 0; i < t; ++i) {
                bit_map[i] = 0;
            }
            for(int i = 0;i<capacity;++i){
                node->array[i].first = 0;
            }
            node_memory_count += memory_size;
            return node;
        }

        inline
        static void
        delete_segment(self_type *node) {
            node_memory_count -= sizeof(self_type) + sizeof(slot_type) * node->capacity +
                                 sizeof(unsigned int) * ((node->capacity >> 5) + 1);
            std::free(node);
        }

        inline int hash(double key) {
            auto predict_hash = (long long) (forward(key) * hash_factor);
            return int(predict_hash % (long long) capacity);
        }

        int find(double key) {
            if (ordered) { return find_ordered(key); }
            auto position = hash(key);
            auto left = position;
            auto right = position;
            for (int i = 0; i <= max_offset ; ++i,--left,++right) {
                if (left < 0) {
                    left += capacity;
                }
                if (right >= capacity) {
                    right -= capacity;
                }
                if (get_bitmap(bitmap_start(), left) && array[left].first == key) {
                    return left;
                }
                if (get_bitmap(bitmap_start(), right) && array[right].first == key) {
                    return right;
                }
            }
            return -1;
        }

        int find_with_cost(double key) {
            if (ordered) {
                ++leaf_cost;
                leaf_max_cost = std::max<long long>(leaf_max_cost,
                                                    (long long) std::log2(double(std::max(1, size))));
                return find_ordered(key);
            }
            auto position = hash(key);
            auto left = position;
            auto right = position;
#ifdef PILOT_DIAGNOSTICS
            // hash leaves must record too, otherwise per-leaf diagnostics only ever cover the
            // ordered population and the two leaf forms cannot be compared on the same leaf
            auto record = [&](int steps) {
                if (leaf_id >= 0) {
                    ++leaf_stats[leaf_id].lookups;
                    leaf_stats[leaf_id].probe += steps;
                }
            };
#else
            auto record = [](int) {};
#endif
            for (int i = 0; i <= max_offset; ++i,--left,++right) {
                if (left < 0) {
                    left += capacity;
                }
                if (right >= capacity) {
                    right -= capacity;
                }
                if (get_bitmap(bitmap_start(), left) && array[left].first == key) {
                    record(i + 1);
                    return left;
                }
                if (get_bitmap(bitmap_start(), right) && array[right].first == key) {
                    record(i + 1);
                    return right;
                }
                ++leaf_cost;
                leaf_max_cost = std::max<long long>(leaf_max_cost,i);
            }
            record(max_offset + 1);
            return -1;
        }

        // Uses the leaf's own min/max keys as a linear model of the rank, then
        // gallops outward from the predicted slot and binary searches the bracket.
        // Accurate prediction (locally uniform keys) -> tiny bracket -> O(1);
        // bad prediction degrades to O(log size).
        int find_ordered(double key) {
            long long steps = 0;
            int result = -1;
            if (size > 0) {
                const double kmin = array[0].first;
                const double kmax = array[size - 1].first;
                if (key >= kmin && key <= kmax) {
                    int p = (kmax > kmin) ? int(double(size - 1) * (key - kmin) / (kmax - kmin)) : 0;
                    if (p < 0) { p = 0; }
                    if (p > size - 1) { p = size - 1; }
                    steps = 1;//the prediction access
                    if (array[p].first == key) {
                        result = p;
                    } else {
                        int lo;
                        int hi;
                        if (array[p].first < key) {//key lies to the right of the prediction
                            int last = p;
                            int d = 1;
                            while (p + d < size && array[p + d].first < key) {
                                last = p + d;
                                d <<= 1;
                                ++steps;
                            }
                            lo = last + 1;
                            hi = std::min(size - 1, p + d);
                        } else {//key lies to the left of the prediction
                            int last = p;
                            int d = 1;
                            while (p - d >= 0 && array[p - d].first > key) {
                                last = p - d;
                                d <<= 1;
                                ++steps;
                            }
                            lo = std::max(0, p - d);
                            hi = last - 1;
                        }
#ifdef PILOT_DIAGNOSTICS
                        ordered_probe_sum += (hi - lo + 1);
                        ++ordered_lookup_count;
#endif
                        while (lo <= hi) {
                            ++steps;
                            int mid = (lo + hi) >> 1;
                            double k = array[mid].first;
                            if (k == key) { result = mid; break; }
                            if (k < key) { lo = mid + 1; } else { hi = mid - 1; }
                        }
                    }
                }
            }
#ifdef PILOT_DIAGNOSTICS
            if (leaf_id >= 0) {
                LeafStat &st = leaf_stats[leaf_id];
                ++st.lookups;
                st.probe += steps;
            }
#else
            (void) steps;
#endif
            return result;
        }

        int find_insert(double key) {
            int position = hash(key);
            int length = (capacity >> 1) + 1;
            int i = 0;
            int left = position;
            int right = position;
            for (; i <= max_offset; ++i,--left,++right) {
                if (left < 0) {
                    left += capacity;
                }
                if (right >= capacity) {
                    right -= capacity;
                }
                if (!get_bitmap(bitmap_start(), left)) {
                    return left;
                } else {
                    if (array[left].first == key) {
                        return -1;
                    }
                }
                if (!get_bitmap(bitmap_start(), right)) {
                    return right;
                } else {
                    if (array[right].first == key) {
                        return -1;
                    }
                }
            }
            for (; i <= length; ++i,--left,++right) {
                if (left < 0) {
                    left += capacity;
                }
                if (right >= capacity) {
                    right -= capacity;
                }
                if (!get_bitmap(bitmap_start(), left)) {
                    max_offset = i;
                    return left;
                }
                if (!get_bitmap(bitmap_start(), right)) {
                    max_offset = i;
                    return right;
                }
            }
            return -1;
        }
    };


    // Pilot: build a leaf either as the original hash leaf or, when the region is cheap to
    // search with a linear model, as a densely packed key-sorted leaf. Criterion:
    //   LEAF_WINDOW_THRESHOLD > 0 : measured p99 prediction error in slots <= threshold
    //   otherwise                 : local skew <= ADAPTIVE_THETA
    template<class key_T, class value_T>
    DataNode<key_T, value_T> *build_leaf_(
            typename std::vector<std::pair<key_T, value_T>>::const_iterator begin,
            typename std::vector<std::pair<key_T, value_T>>::const_iterator end,
            double lower, double upper) {
        using data_node_type = DataNode<key_T, value_T>;
        auto data_count = int(end - begin);
        // Workload-aware load factor: the harness filled workload_model with the query
        // distribution's key histogram before bulk_load, so this leaf's query rate decides whether
        // it is packed (cold) or left at default_density (hot). Empty model -> default_density.
        const double leaf_density = workload_leaf_density(data_count, lower, upper);

        // Compute only what the active criterion needs. With every criterion off (the production
        // default) this function does exactly what the original leaf construction did: no copy,
        // no sort, no extra pass.
        const bool want_skew = (adaptive_theta > 0) || pilot_diagnostics;
        const bool want_window = (leaf_window_threshold > 0) || pilot_diagnostics;
        // the ordered layout itself needs the keys in order, so the copy is also needed when the
        // skew criterion could select it
        const bool want_sorted = want_window || (adaptive_theta > 0) || (leaf_cost_decision > 0);

        double skew = 1.0;
        if (want_skew) { skew = leaf_skew<key_T, value_T>(begin, end, lower, upper); }

        std::vector<std::pair<key_T, value_T>> sorted_keys;
        double window_p99 = 0;
        double window_max = 0;
        if (want_sorted) {
            sorted_keys.assign(begin, end);
            auto by_key = [](const std::pair<key_T, value_T> &a, const std::pair<key_T, value_T> &b) {
                return a.first < b.first;
            };
            // bulk_load requires the caller to have sorted the dataset, so a leaf's slice is
            // normally already ordered and the sort would be wasted work
            if (!std::is_sorted(sorted_keys.begin(), sorted_keys.end(), by_key)) {
                std::sort(sorted_keys.begin(), sorted_keys.end(), by_key);
            }
            if (want_window) {
                leaf_prediction_error<key_T, value_T>(sorted_keys, window_p99, window_max);
            }
#ifdef PILOT_DIAGNOSTICS
            leaf_windows.push_back(window_p99);
#endif
        }
#ifndef PILOT_DIAGNOSTICS
        (void) skew;
#endif

        // ---- two-sided leaf decision (LEAF_COST_DECISION) ----
        // Build a hash leaf, measure its mean point-lookup work per key (leaf_cost), take the
        // ordered leaf's mean work from mean_ordered_steps, and keep the cheaper form under the
        // same (memory + query) weights the agents' reward uses. No window, no threshold constant.
        // The query terms are averaged over the leaf's keys (uniform-over-keys), as elsewhere.
        data_node_type *prebuilt_hash = nullptr;
        bool use_ordered;
        if (leaf_cost_decision > 0) {
            prebuilt_hash = data_node_type::new_segment(
                    std::max(int(float(data_count) / default_density), DATA_NODE_SIZE), lower, upper);
            for (auto ii = begin; ii < end; ++ii) {
                auto position = prebuilt_hash->find_insert(ii->first);
                if (position < 0) { throw MyException("bulk load insertion bad position!"); }
                prebuilt_hash->array[position].first = ii->first;
                prebuilt_hash->array[position].second = ii->second;
                set_bitmap(prebuilt_hash->bitmap_start(), position);
                ++prebuilt_hash->size;
            }
            // Mean slots examined to find a key in the hash leaf: probe outward from the predicted
            // slot, checking left then right. Counts every slot read, including the one holding the
            // key, so it is the same unit as mean_ordered_steps (the prediction access plus each
            // gallop/binary step). find_with_cost omits the successful access, which would make a
            // collision-free hash leaf look free and wrongly win the comparison.
            long long hash_accesses = 0;
            const int hash_capacity = prebuilt_hash->capacity;
            for (auto ii = begin; ii < end; ++ii) {
                const key_T key = ii->first;
                int left = prebuilt_hash->hash(key);
                int right = left;
                for (int i = 0; i <= prebuilt_hash->max_offset; ++i, --left, ++right) {
                    if (left < 0) { left += hash_capacity; }
                    if (right >= hash_capacity) { right -= hash_capacity; }
                    ++hash_accesses;
                    if (get_bitmap(prebuilt_hash->bitmap_start(), left) && prebuilt_hash->array[left].first == key) { break; }
                    ++hash_accesses;
                    if (get_bitmap(prebuilt_hash->bitmap_start(), right) && prebuilt_hash->array[right].first == key) { break; }
                }
            }
            const double mu_h = double(hash_accesses) / double(std::max(1, data_count));
            const double mu_o = mean_ordered_steps<key_T, value_T>(sorted_keys);
            // memory per key, in slots: capacity / size (the slot size in bytes cancels out)
            const double mem_o = double(std::max(data_count, DATA_NODE_SIZE)) / double(std::max(1, data_count));
            const double mem_h = double(prebuilt_hash->capacity) / double(std::max(1, data_count));
            // range-query locate cost, in slots scanned per query: the ordered leaf binary-searches
            // to the lower bound; the hash leaf must scan every slot (its slots are not in key
            // order). LEAF_RANGE_WEIGHT = 0 removes these terms.
            const double range_o = std::log2(std::max(2.0, double(data_count)));
            const double range_h = double(prebuilt_hash->capacity);
            const double cost_o = QUERY_WEIGHT * leaf_cost_weight * mu_o
                                  + leaf_range_weight * leaf_cost_weight * range_o
                                  + MEMORY_WEIGHT * mem_o;
            const double cost_h = QUERY_WEIGHT * leaf_cost_weight * mu_h
                                  + leaf_range_weight * leaf_cost_weight * range_h
                                  + MEMORY_WEIGHT * mem_h;
            use_ordered = (cost_o < cost_h) && (data_count > DATA_NODE_SIZE);
            if (data_count > 1000000) {
                fprintf(stderr, "COSTDEBUG size=%d mu_o=%.3f mu_h=%.3f mem_o=%.3f mem_h=%.3f cost_o=%.3f cost_h=%.3f ordered=%d\n",
                        data_count, mu_o, mu_h, mem_o, mem_h, cost_o, cost_h, int(cost_o < cost_h));
            }
        } else if (leaf_window_threshold > 0) {
            use_ordered = (window_p99 <= leaf_window_threshold) && (data_count > DATA_NODE_SIZE);
        } else {
            use_ordered = (adaptive_theta > 0) && (skew <= adaptive_theta) && (data_count > DATA_NODE_SIZE);
        }

#ifdef PILOT_DIAGNOSTICS
        // leaf_skew divides by (upper - lower) while the linear model spans the keys actually
        // present; record both so a real skew/window decoupling can be told from a normalization
        // artifact
        double key_span = 0;
        if (data_count > 0) {
            double key_min = begin->first;
            double key_max = begin->first;
            for (auto it = begin; it != end; ++it) {
                key_min = std::min(key_min, double(it->first));
                key_max = std::max(key_max, double(it->first));
            }
            key_span = key_max - key_min;
        }
        const double interval_width = upper - lower;
        ++leaf_count;
        leaf_skews.push_back(skew);
#endif

        if (use_ordered) {
            if (prebuilt_hash != nullptr) { data_node_type::delete_segment(prebuilt_hash); }
#ifdef PILOT_DIAGNOSTICS
            ++ordered_leaf_count;
#endif
            auto node = data_node_type::new_segment(std::max(data_count, DATA_NODE_SIZE), lower, upper);
            for (int i = 0; i < data_count; ++i) {
                node->array[i] = sorted_keys[i];
                set_bitmap(node->bitmap_start(), i);
            }
            node->size = data_count;
            node->ordered = 1;
#ifdef PILOT_DIAGNOSTICS
            node->leaf_id = int(leaf_stats.size());
            leaf_stats.push_back({data_count, 1, skew, 0, 0, window_p99, window_max, interval_width, key_span, lower, upper});
#endif
            return node;
        }
        // LEAF_COST_DECISION measured and kept the hash leaf; reuse it instead of rebuilding
        if (prebuilt_hash != nullptr) {
#ifdef PILOT_DIAGNOSTICS
            prebuilt_hash->leaf_id = int(leaf_stats.size());
            leaf_stats.push_back({data_count, 0, skew, 0, 0, window_p99, window_max, interval_width, key_span, lower, upper});
#endif
            return prebuilt_hash;
        }
        auto node = data_node_type::new_segment(
                std::max(int(float(data_count) / leaf_density), DATA_NODE_SIZE), lower, upper);
        for (auto ii = begin; ii < end; ++ii) {
            auto position = node->find_insert(ii->first);
            if (position < 0) { throw MyException("bulk load insertion bad position!"); }
            node->array[position].first = ii->first;
            node->array[position].second = ii->second;
            set_bitmap(node->bitmap_start(), position);
            ++node->size;
        }
#ifdef PILOT_DIAGNOSTICS
        node->leaf_id = int(leaf_stats.size());
        leaf_stats.push_back({data_count, 0, skew, 0, 0, window_p99, window_max, interval_width, key_span, lower, upper});
#endif
        return node;
    }


    template<class key_T, class value_T>
    class InnerNode {
    public:
        struct Slot {
            union {
                InnerNode *inner_node;
                DataNode<key_T, value_T> *data_node;
            };
        };
        using self_type = InnerNode;
        using slot_type = Slot;
    public:
        double lower{0};
        double upper{0};
        int capacity{-1};

        [[nodiscard]] inline auto forward(double key) const {
            return std::min(int(capacity - 1), std::max(0, int(forward_with_parameters(lower, upper, capacity, key))));
        }

        [[nodiscard]] inline
        std::pair<double, double> sub_interval(int position) const {
            auto slot_interval = (upper - lower) / double(capacity);
            return {slot_interval * double(position) + lower,
                    slot_interval * double(position + 1) + lower};
        }

        ///////
        slot_type array[0];

        inline unsigned int *bitmap_start() {
            return (unsigned int *) (array + capacity);
        }

        inline
        static self_type *
        new_segment(int capacity, double lower, double upper) {
            int t = ((capacity >> 5) + 1);
            auto memory_size = sizeof(self_type) + sizeof(slot_type) * capacity + sizeof(unsigned int) * t;
            auto node = (self_type *) std::malloc(memory_size);
            node->capacity = capacity;
            node->lower = lower;
            node->upper = upper;
            auto bit_map = node->bitmap_start();
            for (int i = 0; i < t; ++i) {
                bit_map[i] = 0;
            }
            node_memory_count += memory_size;
            return node;
        }

        inline
        static void
        delete_segment(self_type *node) {
            node_memory_count -= sizeof(self_type) + sizeof(slot_type) * node->capacity +
                                 sizeof(unsigned int) * ((node->capacity >> 5) + 1);
            std::free(node);
        }
    };

    class rebuild_lock {
        std::mutex position_lock;
        int frontend_root = -1;
        int frontend_inner = -1;
        int backend_root = -1;
        int backend_inner = -1;
    public:
        void set_frontend_position(int root, int inner) {
            while (true) {
                position_lock.lock();
                if (root != backend_root || inner != backend_inner) {
                    frontend_root = root;
                    frontend_inner = inner;
                    position_lock.unlock();
                    break;
                }
                position_lock.unlock();
            }
        }

        void set_backend_position(int root, int inner) {
            while (true) {
                position_lock.lock();
                if (root != frontend_root || inner != frontend_inner) {
                    backend_root = root;
                    backend_inner = inner;
                    position_lock.unlock();
                    break;
                }
                position_lock.unlock();
            }
        }

        void de_set_frontend_position() {
            position_lock.lock();
            frontend_root = -1;
            frontend_inner = -1;
            position_lock.unlock();
        }

        void de_set_backend_position() {
            position_lock.lock();
            backend_root = -1;
            backend_inner = -1;
            position_lock.unlock();
        }
    };

    template<class key_T, class value_T>
    std::vector<std::pair<std::pair<int, int>, std::pair<double, double>>> split_dataset(
            typename std::vector<std::pair<key_T, value_T>>::const_iterator begin,
            typename std::vector<std::pair<key_T, value_T>>::const_iterator end,
            Hits::InnerNode<key_T, value_T> *inner_node) {
        std::vector<std::pair<double, double>> intervals;
        std::vector<int> bucket(inner_node->capacity);
        for (auto i = begin; i < end; ++i) {
            ++bucket[inner_node->forward(i->first)];
        }
        std::vector<std::pair<std::pair<int, int>, std::pair<double, double>>> result;
        int left = 0, right = 0;
        for (int i = 0; i < inner_node->capacity; ++i) {
            left = right;
            right += bucket[i];
            result.push_back({{left, right}, inner_node->sub_interval(i)});
        }
        return result;
    }
    // ---- range query: walk the tree in key order, collect keys in [lo, hi] ----
    // An inner node's children are ordered by key interval (array index == key order), so the
    // walk can skip subtrees entirely below the range and stop at the first one above it.
    template<class key_T, class value_T>
    void range_collect_leaf(DataNode<key_T, value_T> *leaf, key_T lo, key_T hi,
                            std::vector<std::pair<key_T, value_T>> &out) {
        if (leaf->ordered) {
            // keys are packed in slots [0, size) in sorted order, so the lower bound is a
            // binary search and the scan then stops at the first key above the range
            auto first = leaf->array;
            auto last = leaf->array + leaf->size;
            auto it = std::lower_bound(first, last, lo,
                                       [](const std::pair<key_T, value_T> &a, key_T v) {
                                           return a.first < v;
                                       });
            for (; it != last && it->first <= hi; ++it) { out.push_back(*it); }
            return;
        }
        // hash leaf: slot order is not key order, so every slot must be examined and a slot
        // above the range cannot terminate the scan
        for (int i = 0; i < leaf->capacity; ++i) {
            if (!get_bitmap(leaf->bitmap_start(), i)) { continue; }
            key_T k = leaf->array[i].first;
            if (k < lo || k > hi) { continue; }
            out.push_back(leaf->array[i]);
        }
    }

    template<class key_T, class value_T>
    void range_collect_node(InnerNode<key_T, value_T> *node, key_T lo, key_T hi,
                            std::vector<std::pair<key_T, value_T>> &out) {
        // Children are equal-width and key-ordered, so interpolation (the same forward() the point
        // lookup uses) lands on the first child whose interval can reach lo; scanning children from
        // index 0 would make the walk O(fanout) per level instead of O(1) + the touched children.
        // One step back guards against a rounding off-by-one.
        int first = std::max(0, node->forward(lo) - 1);
        for (int i = first; i < node->capacity; ++i) {
            auto interval = node->sub_interval(i);
            if (interval.second <= lo) { continue; }//entirely below the range
            if (interval.first > hi) { break; }     //ordered children: nothing above can match
            if (get_bitmap(node->bitmap_start(), i)) {
                range_collect_node<key_T, value_T>(node->array[i].inner_node, lo, hi, out);
            } else {
                range_collect_leaf<key_T, value_T>(node->array[i].data_node, lo, hi, out);
            }
        }
    }

    template<class key_T, class value_T>
    class Index {
    public:
        using self_type = Index<key_T, value_T>;
        using inner_node_type = InnerNode<key_T, value_T>;
        using data_node_type = DataNode<key_T, value_T>;

        double lower{};
        double upper{};
        Configuration conf;
        inner_node_type *root{};
        bool is_leaf{false};
#ifdef CB
        int rebuild_step = 0;//just a bool
        rebuild_lock lock;
#endif
#ifdef using_small_network
        std::shared_ptr<Small_Q_network> q_network;
#endif
    public:

        float memory_occupied() {
            return ((float) node_memory_count + sizeof(self_type));
        }

        std::pair<bool, void *> build_(typename std::vector<std::pair<key_T, value_T>>::const_iterator begin,
                                       typename std::vector<std::pair<key_T, value_T>>::const_iterator end,
                                       std::pair<double, double> interval) {//is inner / the pointer of node
#ifndef using_small_network
            auto data_node = DataNode<key_T,value_T>::new_segment(
                    std::max(int(float(end - begin)/default_density),DATA_NODE_SIZE),
                    interval.first,interval.second);
            for(auto i = begin;i<end;++i){
                auto slot_data = *i;
                auto position = data_node->find_insert(slot_data.first);
                if(position < 0){
                    throw MyException("bulk load insertion bad position!");
                }
                data_node->array[position] = slot_data;
                set_bitmap(data_node->bitmap_start(),position);
                ++data_node->size;
            }
            return {false,data_node};
#else
            auto pointer_result = std::pair<bool, void *>({false, nullptr});
            auto data_count = int(end - begin);
            int fanout = 1;
            if (target_leaf > 0) {
                if (data_count > target_leaf) { fanout = target_fanout; }
            } else if (data_count > DATA_NODE_SIZE) {
                auto pdf = get_pdf<key_T, value_T>(begin, end, interval.first, interval.second, SMALL_PDF_SIZE);
                auto a = torch::tensor(pdf).view({1, SMALL_PDF_SIZE}).mul(
                        torch::ones({int(action_space.size()), SMALL_PDF_SIZE})).to(GPU_DEVICE);
                auto b = torch::tensor(data_count).view({1, 1}).mul(torch::ones({int(action_space.size()), 1})).to(GPU_DEVICE);
                auto c = torch::tensor(action_space).to(torch::kFloat32).view({int(action_space.size()), 1}).to(GPU_DEVICE);
                auto pred_c = q_network->forward(a, b, c).to(CPU_DEVICE);
                auto ac_id = torch::argmax(pred_c,0).item().toInt();
                fanout = 1 << ac_id;
            }
            if (fanout > 1) {
                pointer_result.first = true;
                //split dataset
                //establish a inner node with capacity as fanout  // model based without establish physically
                auto inner_node = Hits::InnerNode<key_T, value_T>::new_segment(fanout, interval.first,interval.second);
                pointer_result.second = inner_node;
                std::vector<std::pair<std::pair<int, int>, std::pair<double, double>>> result = split_dataset<key_T, value_T>(begin, end, inner_node);
                assert(result.size() == inner_node->capacity);
                for (int ii = 0; ii < int(result.size()); ++ii) {
                    auto tmp_result = build_(begin + result[ii].first.first,
                                             begin + result[ii].first.second, result[ii].second);
                    if (tmp_result.first) {
                        set_bitmap(inner_node->bitmap_start(), ii);
                        inner_node->array[ii].inner_node = (InnerNode<key_T, value_T> *) tmp_result.second;
                    } else {
                        de_set_bitmap(inner_node->bitmap_start(), ii);
                        inner_node->array[ii].data_node = (DataNode<key_T, value_T> *) tmp_result.second;
                    }
                }
            } else {
                // leaf: ordered-dense when the region is not locally skewed, hash otherwise
                pointer_result.second = build_leaf_<key_T, value_T>(begin, end, interval.first, interval.second);
            }
            return pointer_result;
#endif
        }

        void bulk_load(
                typename std::vector<std::pair<key_T,value_T>>::const_iterator begin,
                typename std::vector<std::pair<key_T,value_T>>::const_iterator end){
#ifdef CB
            stop_rebuild();
            rebuild_step = 0;
#endif
            delete_tree(root);
            inner_node_type::delete_segment(root);
            root = inner_node_type::new_segment(this->conf.root_fan_out, lower, upper);
            auto root_result = split_dataset(begin,end,root);
            for (int root_slot_id = 0; root_slot_id < root->capacity; ++root_slot_id) {
                auto inner_fanout = get_fanout(0,(root_result[root_slot_id].second.first + root_result[root_slot_id].second.second)/2);
                InnerNode<key_T,value_T>*  inner_node = inner_node_type::new_segment(inner_fanout, root_result[root_slot_id].second.first, root_result[root_slot_id].second.second);
                root->array[root_slot_id].inner_node = inner_node;
                set_bitmap(root->bitmap_start(),root_slot_id);
                auto inner_result = split_dataset(begin + root_result[root_slot_id].first.first,begin + root_result[root_slot_id].first.second,inner_node);
                for (int inner_slot_id = 0; inner_slot_id < inner_node->capacity; ++inner_slot_id) {
#ifdef using_small_network
                    auto build_result = build_(begin + root_result[root_slot_id].first.first + inner_result[inner_slot_id].first.first,
                                               begin + root_result[root_slot_id].first.first + inner_result[inner_slot_id].first.second,
                                               {inner_result[inner_slot_id].second.first, inner_result[inner_slot_id].second.second});
                    if (build_result.first) {
                        set_bitmap(inner_node->bitmap_start(), inner_slot_id);
                        inner_node->array[inner_slot_id].inner_node = (InnerNode<key_T, value_T> *) build_result.second;
                    } else {
                        de_set_bitmap(inner_node->bitmap_start(), inner_slot_id);
                        inner_node->array[inner_slot_id].data_node = (DataNode<key_T, value_T> *) build_result.second;
                    }
#endif
#ifndef using_small_network
                    auto data_node = build_leaf_<key_T, value_T>(
                            begin + root_result[root_slot_id].first.first + inner_result[inner_slot_id].first.first,
                            begin + root_result[root_slot_id].first.first + inner_result[inner_slot_id].first.second,
                            inner_result[inner_slot_id].second.first, inner_result[inner_slot_id].second.second);
                    de_set_bitmap(inner_node->bitmap_start(), inner_slot_id);
                    inner_node->array[inner_slot_id].data_node = data_node;
#endif
                }
            }
#ifdef CB
            start_rebuild();
#endif
        }

#ifdef CB
    private:
        void stop_rebuild() {
            auto status = rebuild_step;
            ++rebuild_step;
            while (status + 1 == rebuild_step) { sleep(1); }
        }

        void start_rebuild() {
            rebuild_step = 0;
            auto status = rebuild_step;
            std::thread(std::bind(&Index::random_rebuild, this)).detach();
            while (status == rebuild_step);
        }

        std::vector<typename DataNode<key_T, value_T>::slot_type> dataset_for_rebuild;

        void get_data(InnerNode<key_T, value_T> *node) {
            for (int i = 0; i < node->capacity; ++i) {
                if (get_bitmap(node->bitmap_start(), i)) {
                    //inner node need to do later
                    get_data(node->array[i].inner_node);
                } else {//data needed to store in to vector
                    auto data_node = node->array[i].data_node;
                    for (int j = 0; j < data_node->capacity; ++j) {
                        if (get_bitmap(data_node->bitmap_start(), j)) {
                            dataset_for_rebuild.push_back(data_node->array[j]);
                        }
                    }
                }
            }
        }

        void random_rebuild() {
            ++rebuild_step;
            auto status = rebuild_step;
//            puts("rebuild start!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
            while (true) {
//                sleep(1);
//                if (status != rebuild_step) {
//                    ++rebuild_step;
////                            puts("rebuild stop!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
//                    return;
//                }
                puts("rebuild!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
                for (int root_slot_id = 0; root_slot_id < root->capacity; ++root_slot_id) {
                    auto inner_node = root->array[root_slot_id].inner_node;
                    for (int inner_slot_id = 0; inner_slot_id < inner_node->capacity; ++inner_slot_id) {
                        lock.set_backend_position(root_slot_id, inner_slot_id);
                        auto min_max = std::pair<double, double>();
                        dataset_for_rebuild.clear();
                        auto data_node = inner_node->array[inner_slot_id].data_node;
                        min_max = {data_node->lower, data_node->upper};
                        for (int k = 0; k < data_node->capacity; ++k) {
                            if (get_bitmap(data_node->bitmap_start(), k)) {
                                dataset_for_rebuild.push_back(data_node->array[k]);
                            }
                        }
                        DataNode<key_T, value_T>::delete_segment(data_node);
                        data_node = DataNode<key_T,value_T>::new_segment(
                                std::max(int(float(dataset_for_rebuild.size())/default_density),DATA_NODE_SIZE),
                                min_max.first,min_max.second);
                        for(auto &i:dataset_for_rebuild){
                            auto position = data_node->find_insert(i.first);
                            if(position < 0){
                                throw MyException("bulk load insertion bad position!");
                            }
                            data_node->array[position] = i;
                            set_bitmap(data_node->bitmap_start(),position);
                            ++data_node->size;
                        }
                        de_set_bitmap(inner_node->bitmap_start(), inner_slot_id);
                        inner_node->array[inner_slot_id].data_node = data_node;
                        lock.de_set_backend_position();
                        if (status != rebuild_step) {
                            ++rebuild_step;
//                            puts("rebuild stop!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
                            return;
                        }
                    }
                    usleep(sleep_time);
//                    usleep(11190);
//                    sleep(1);
                }
            }
        }
#endif
    public:
        Index(Configuration conf, double lower, double upper) : conf(conf) {
            this->lower = lower;
            this->upper = upper;
            root = inner_node_type::new_segment(this->conf.root_fan_out, lower, upper);
            for (int i = 0; i < root->capacity; ++i) {
                auto child_interval = root->sub_interval(i);
                auto fanout = get_fanout(0, (child_interval.first + child_interval.second) / 2);
                auto new_inner_node = inner_node_type::new_segment(fanout, child_interval.first, child_interval.second);
                root->array[i].inner_node = new_inner_node;//data node || no need for set bitmap
                set_bitmap(root->bitmap_start(), i);
                for (int j = 0; j < new_inner_node->capacity; ++j) {
                    auto data_child_interval = new_inner_node->sub_interval(j);
                    auto new_data_node = data_node_type::new_segment(DATA_NODE_SIZE, data_child_interval.first,
                                                                     data_child_interval.second);
                    new_inner_node->array[j].data_node = new_data_node;//data node || no need for set bitmap
                    de_set_bitmap(new_inner_node->bitmap_start(), j);
                }
            }
#ifdef using_small_network
            q_network = std::make_shared<Small_Q_network>();
            torch::load(q_network, model_father_path + "tmp.pt");
            q_network->to(GPU_DEVICE);
            q_network->eval();
#endif
#ifdef CB
            start_rebuild();
#endif
        }

        ~Index() {
#ifdef CB
            stop_rebuild();
#endif
            if (is_leaf) {
                data_node_type::delete_segment((DataNode<key_T, value_T> *) root);
                return;
            }
            delete_tree(root);
            inner_node_type::delete_segment(root);
            root = nullptr;
        }


        inline int get_fanout(int layer, double node_interval_point) {
            auto pred = std::min(double(INNER_FANOUT_COLUMN - 1),
                                 std::max(0.0, double(INNER_FANOUT_COLUMN - 1) *
                                               (node_interval_point - lower) / (upper - lower)));
            auto left = int(pred);
            pred -= left;
            return int(0.5 + this->conf.fan_outs[std::min(INNER_FANOUT_ROW - 1, layer)][left] * (1 - pred) +
                       this->conf.fan_outs[std::min(INNER_FANOUT_ROW - 1, layer)][left + 1] * pred);
        }

        // Collect every entry whose key lies in [lo, hi]. The result is in key order for ordered
        // leaves and in slot order for hash leaves, so a caller comparing the two leaf forms must
        // treat it as a multiset.
        void range_query(key_T lo, key_T hi, std::vector<std::pair<key_T, value_T>> &out) const {
            out.clear();
            range_collect_node<key_T, value_T>(root, lo, hi, out);
        }

        bool add(key_T key, const value_T &value) {

            auto root_position = root->forward(key);
            auto node = root->array[root_position].inner_node;
            auto position = node->forward(key);
#ifdef CB
            lock.set_frontend_position(root_position, position);
#endif
            while (get_bitmap(node->bitmap_start(), position)) {//true first_moment there is a pointer of a inner node
                node = node->array[position].inner_node;
                position = node->forward(key);
            }
            auto data_node = node->array[position].data_node;
            if (data_node->size > data_node->capacity * max_density) {
#ifdef count_insert_time
                tc.synchronization();
#endif
                //expand node
                DataNode<key_T, value_T> *new_data_node = data_node_type::new_segment(
                        data_node->size * (1.0 / default_density), data_node->lower, data_node->upper);
                for (int i = 0; i < data_node->capacity; ++i) {
                    if (get_bitmap(data_node->bitmap_start(), i)) {
                        auto new_position = new_data_node->find_insert(data_node->array[i].first);
                        new_data_node->array[new_position] = data_node->array[i];
                        set_bitmap(new_data_node->bitmap_start(), new_position);
                        ++new_data_node->size;
                    }
                }
                data_node_type::delete_segment(data_node);
                node->array[position].data_node = new_data_node;
                data_node = new_data_node;
#ifdef count_insert_time
                insert_time_retrain += tc.get_timer_second();
#endif
            }
            auto data_position = data_node->find_insert(key);
            if (data_position < 0) {
#ifdef CB
                lock.de_set_frontend_position();
#endif
                return false;
            }
            data_node->array[data_position].first = key;
            data_node->array[data_position].second = value;
            set_bitmap(data_node->bitmap_start(), data_position);
            ++data_node->size;
#ifdef CB
            lock.de_set_frontend_position();
#endif
            return true;
        }

        bool get(key_T key, value_T &value) {

            auto root_position = root->forward(key);
            auto node = root->array[root_position].inner_node;
            auto position = node->forward(key);
#ifdef CB
            lock.set_frontend_position(root_position, position);
#endif
            //to find a data node (leaf node)
            while (get_bitmap(node->bitmap_start(), position)) {//true first_moment there is a pointer of a inner node
                node = node->array[position].inner_node;
                position = node->forward(key);
            }
            auto data_node = node->array[position].data_node;
            if (runtime_adaptive) { ++data_node->hits; }
            auto data_position = data_node->find(key);
            if (data_position < 0 || data_node->array[data_position].first != key) {//offset == max_offset
#ifdef CB
                lock.de_set_frontend_position();
#endif
                return false;
            } else {
                value = data_node->array[data_position].second;
#ifdef CB
                lock.de_set_frontend_position();
#endif
                return true;
            }
        }

        // ---- runtime adaptation (RUNTIME_ADAPTIVE) ----
        void collect_leaves_(InnerNode<key_T, value_T> *node,
                             std::vector<DataNode<key_T, value_T> *> &leaves,
                             std::vector<InnerNode<key_T, value_T> *> &parents,
                             std::vector<int> &slots) {
            for (int i = 0; i < node->capacity; ++i) {
                if (get_bitmap(node->bitmap_start(), i)) {
                    collect_leaves_(node->array[i].inner_node, leaves, parents, slots);
                } else {
                    leaves.push_back(node->array[i].data_node);
                    parents.push_back(node);
                    slots.push_back(i);
                }
            }
        }

        // Re-decide every hash leaf's density from its observed access count, with no build-time
        // workload model. A leaf whose per-key rate has crossed the global mean (with a +-10%
        // hysteresis band) is rebuilt at the other density. Counters are halved first so the
        // statistics track recent traffic rather than all history. Single-threaded: no locking.
        // Returns the number of leaves rebuilt.
        long long rebalance_leaves() {
            if (!runtime_adaptive || root == nullptr) { return 0; }
            std::vector<DataNode<key_T, value_T> *> leaves;
            std::vector<InnerNode<key_T, value_T> *> parents;
            std::vector<int> slots;
            collect_leaves_(root, leaves, parents, slots);
            for (auto *l: leaves) { l->hits >>= 1; }
            double total_hits = 0, total_keys = 0;
            for (auto *l: leaves) { total_hits += double(l->hits); total_keys += double(l->size); }
            if (!(total_hits > 0) || !(total_keys > 0)) { return 0; }
            const double avg = total_hits / total_keys;
            long long flipped = 0;
            for (size_t k = 0; k < leaves.size(); ++k) {
                auto *l = leaves[k];
                if (l->ordered || l->size <= 0) { continue; }// hash leaves only
                const int cap_hot = std::max(int(float(l->size) / default_density), DATA_NODE_SIZE);
                const int cap_cold = std::max(int(float(l->size) / workload_cold_density), DATA_NODE_SIZE);
                if (cap_cold == cap_hot) { continue; }// too small for the two densities to differ
                const bool is_cold = (l->capacity == cap_cold);
                const double rate = double(l->hits) / double(l->size);
                bool want_cold = is_cold;
                if (!is_cold && rate < 0.9 * avg) { want_cold = true; }
                else if (is_cold && rate > 1.1 * avg) { want_cold = false; }
                if (want_cold == is_cold) { continue; }
                auto *nn = DataNode<key_T, value_T>::new_segment(want_cold ? cap_cold : cap_hot,
                                                                 l->lower, l->upper);
                for (int i = 0; i < l->capacity; ++i) {
                    if (get_bitmap(l->bitmap_start(), i)) {
                        auto p = nn->find_insert(l->array[i].first);
                        if (p < 0) { throw MyException("rebalance insertion bad position!"); }
                        nn->array[p] = l->array[i];
                        set_bitmap(nn->bitmap_start(), p);
                        ++nn->size;
                    }
                }
                parents[k]->array[slots[k]].data_node = nn;
                DataNode<key_T, value_T>::delete_segment(l);
                ++flipped;
            }
            return flipped;
        }

//        void range_query_(InnerNode<key_T,value_T> *node,std::vector<std::pair<key_T,value_T>> &result,double upper_bound){
//            for(int j = 0;j < node->capacity;++j){
//                //to find a data node (leaf node)
//                if(get_bitmap(node->bitmap_start(), j)) {
//                    range_query_(result->array[j].inner_node,result);
//                }
//                else{
//                    auto data_node = result->array[j].data_node;
//                    for(auto l = 0;l < data_node->capacity;++l) {
//                        if(get_bitmap(data_node->bitmap_start(),l)){
//                            if(data_node->array[l].first < upper_bound) {
//                                result.push_back(data_node->array[l]);
//                            }
//                        }
//                    }
//                }
//                if(result->sub_interval(j).second >= upper_bound){
//                    return;
//                }
//            }
//        }
//
//        std::vector<std::pair<key_T,value_T>> range_query(key_T lower_bound, key_T upper_bound) {
//            std::vector<std::pair<key_T,value_T>> result;
//            auto key = lower_bound;
//            auto root_position = root->forward(key);
//            auto node = root->array[root_position].inner_node;
//            auto position = node->forward(key);
//            for(int i = root_position;i<root->capacity;++i){
//                auto sub_root = root->array[i].inner_node;
//                for(int j = position;j < sub_root->capacity;++j){
//#ifdef CB
//                    lock.set_frontend_position(i, j);
//#endif
                    //to find a data node (leaf node)
//                    if(get_bitmap(node->bitmap_start(), position)) {
//                        range_query_(sub_root->array[j].inner_node,result,upper_bound);
//                    }
//                    else{
//                        auto data_node = sub_root->array[j].data_node;
//                        for(auto l = 0;l < data_node->capacity;++l) {
//                            if(get_bitmap(data_node->bitmap_start(),l)){
//                                if(data_node->array[l].first < upper_bound) {
//                                    result.push_back(data_node->array[l]);
//                                }
//                            }
//                        }
//                    }
//#ifdef CB
//                    lock.de_set_frontend_position();
//#endif
//                    if(sub_root->sub_interval(j).second >= upper_bound){
//                        return result;
//                    }
//                }
//                if(root->sub_interval(i).second >= upper_bound){
//                    return result;
//                }
//            }
//            return result;
//        }


        bool get_with_root_leaf(key_T key, value_T &value) {
            if (is_leaf) {
                auto data_node = (DataNode<key_T, value_T> *) root;
                ++inner_cost;
                auto data_position = data_node->find_with_cost(key);
                if (data_position < 0 || data_node->array[data_position].first != key) {
                    return false;
                } else {
                    value = data_node->array[data_position].second;
                    return true;
                }
            }
            auto node = root;
            auto position = node->forward(key);
            ++inner_cost;
            //to find a data node (leaf node)
            while (get_bitmap(node->bitmap_start(), position)) {//true first_moment there is a pointer of a inner node
                node = node->array[position].inner_node;
                position = node->forward(key);
                ++inner_cost;
            }
            auto data_node = node->array[position].data_node;
            ++inner_cost;
            auto data_position = data_node->find_with_cost(key);
            if (data_position < 0 || data_node->array[data_position].first != key) {
                return false;
            } else {
                value = data_node->array[data_position].second;
                return true;
            }
        }

        bool get_with_cost(key_T key, value_T &value) {
            auto node = root;
            auto position = node->forward(key);
            ++inner_cost;
            //to find a data node (leaf node)
            while (get_bitmap(node->bitmap_start(), position)) {//true first_moment there is a pointer of a inner node
                node = node->array[position].inner_node;
                position = node->forward(key);
                ++inner_cost;
            }
            auto data_node = node->array[position].data_node;
            ++inner_cost;
            auto data_position = data_node->find_with_cost(key);
            if (data_position < 0 || data_node->array[data_position].first != key) {
                return false;
            } else {
                value = data_node->array[data_position].second;
                return true;
            }
        }
        //count the nodes num of each later
        std::vector<long long> count_node_of_each_layer(){
            std::vector<long long> result;
            std::vector<inner_node_type*> q1;
            std::vector<inner_node_type*> q2;
            q1.push_back(root);
            while(true) {
                long long this_layer_count = 0;
                for(auto n:q1) {
                    this_layer_count += n->capacity;
                    for(int i = 0;i<n->capacity;++i) {
                        if(get_bitmap(n->bitmap_start(),i)) {
                            q2.push_back(n->array[i].inner_node);
                        }
                    }
                }
                result.push_back(this_layer_count);
                if(q2.empty()) {
                    break;
                }else {
                    q1 = q2;
                    q2.clear();
                }
            }
            return result;
        }

        bool erase(key_T key) {
            auto root_position = root->forward(key);
            auto node = root->array[root_position].inner_node;
            auto position = node->forward(key);
#ifdef CB
            lock.set_frontend_position(root_position, position);
#endif
            //to find a data node (leaf node)
            while (get_bitmap(node->bitmap_start(), position)) {//true first_moment there is a pointer of a inner node
                node = node->array[position].inner_node;
                position = node->forward(key);
            }
            auto data_node = node->array[position].data_node;
            if (data_node->capacity > DATA_NODE_SIZE && data_node->size < data_node->capacity * min_density) {
                //split node
                data_node_type *new_data_node = data_node_type::new_segment(
                        std::max(int(data_node->size * (1.0 / default_density)),DATA_NODE_SIZE),
                        data_node->lower, data_node->upper);
                for (int i = 0; i < data_node->capacity; ++i) {
                    if (get_bitmap(data_node->bitmap_start(), i)) {
                        auto new_position = new_data_node->find_insert(data_node->array[i].first);
                        new_data_node->array[new_position] = data_node->array[i];
                        set_bitmap(new_data_node->bitmap_start(), new_position);
                        ++new_data_node->size;
                    }
                }
                data_node_type::delete_segment(data_node);
                node->array[position].data_node = new_data_node;
                data_node = new_data_node;
            }
            auto data_position = data_node->find(key);
            if (data_position < 0 || data_node->array[data_position].first != key) {
#ifdef CB
                lock.de_set_frontend_position();
#endif
                return false;
            }
            de_set_bitmap(data_node->bitmap_start(), data_position);
            --data_node->size;
#ifdef CB
            lock.de_set_frontend_position();
#endif
            return true;
        }

        void delete_tree(InnerNode<key_T, value_T> *node) {
            for (int i = 0; i < node->capacity; ++i) {
                if (get_bitmap(node->bitmap_start(), i)) {
                    delete_tree(node->array[i].inner_node);
                    inner_node_type::delete_segment(node->array[i].inner_node);
                } else {
                    data_node_type::delete_segment(node->array[i].data_node);
                }
            }
        }
    };
}

#endif //INDEX_H
