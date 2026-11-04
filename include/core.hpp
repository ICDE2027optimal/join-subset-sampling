// Weighted join index and PRODUCT independent-subset sampler.
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

using Clock = std::chrono::steady_clock;
using Key = std::int64_t;
using Output = std::vector<int>;

inline constexpr Key ROOT_KEY = 0;

struct VecHash {
    std::size_t operator()(const Output& v) const noexcept {
        std::uint64_t h = 0x9e3779b97f4a7c15ULL;
        for (int x : v) {
            std::uint64_t z =
                static_cast<std::uint64_t>(static_cast<std::uint32_t>(x));
            z += 0x9e3779b97f4a7c15ULL;
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
            z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
            z ^= z >> 31;
            h ^= z + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        }
        return static_cast<std::size_t>(h);
    }
};

struct Tuple {
    int id = -1;
    double weight = 0.0;
    std::vector<Key> edge_key;
};

struct Relation {
    std::vector<Tuple> tuples;
};

struct Instance {
    int m = 0;
    std::vector<int> parent;
    std::vector<std::vector<int>> children;
    std::vector<int> preorder;
    std::vector<Relation> rel;
    std::string shape;
    std::string weight_mode;
};

using EdgeId = std::int32_t;

struct UndirectedJoinEdge {
    EdgeId id = -1;
    int left_relation = -1;
    int right_relation = -1;
};

struct UndirectedTuple {
    int id = -1;
    double weight = 0.0;
    std::vector<std::pair<EdgeId, Key>> incident_keys;
};

struct UndirectedRelation {
    std::vector<UndirectedTuple> tuples;
};

struct UndirectedInstance {
    int m = 0;
    std::vector<UndirectedJoinEdge> edges;
    std::vector<UndirectedRelation> rel;
    std::string shape;
    std::string weight_mode;
};

inline void validate_instance(const Instance& in) {
    if (in.m <= 0) throw std::invalid_argument("instance must have m>0");
    const std::size_t m = static_cast<std::size_t>(in.m);
    if (in.parent.size() != m || in.children.size() != m ||
        in.preorder.size() != m || in.rel.size() != m)
        throw std::invalid_argument("instance relation-tree dimensions mismatch m");

    std::vector<int> preorder_position(m, -1);
    for (std::size_t i = 0; i < m; ++i) {
        const int v = in.preorder[i];
        if (v < 0 || v >= in.m || preorder_position[static_cast<std::size_t>(v)] >= 0)
            throw std::invalid_argument("preorder must be a permutation of relations");
        preorder_position[static_cast<std::size_t>(v)] = static_cast<int>(i);
    }

    int root = -1;
    std::vector<int> child_occurrences(m, 0);
    for (int v = 0; v < in.m; ++v) {
        const int parent = in.parent[static_cast<std::size_t>(v)];
        if (parent < 0) {
            if (parent != -1 || root >= 0)
                throw std::invalid_argument("instance must have exactly one -1 root");
            root = v;
        } else if (parent >= in.m || parent == v) {
            throw std::invalid_argument("invalid parent relation index");
        }
        for (int child : in.children[static_cast<std::size_t>(v)]) {
            if (child < 0 || child >= in.m || child == v)
                throw std::invalid_argument("invalid child relation index");
            if (in.parent[static_cast<std::size_t>(child)] != v)
                throw std::invalid_argument("children and parent arrays disagree");
            if (++child_occurrences[static_cast<std::size_t>(child)] != 1)
                throw std::invalid_argument("duplicate child relation");
        }
    }
    if (root < 0 || in.preorder.front() != root)
        throw std::invalid_argument("preorder must start at the unique root");
    for (int v = 0; v < in.m; ++v) {
        if (v == root) {
            if (child_occurrences[static_cast<std::size_t>(v)] != 0)
                throw std::invalid_argument("root appears as a child");
        } else {
            if (child_occurrences[static_cast<std::size_t>(v)] != 1)
                throw std::invalid_argument("non-root must appear once as a child");
            const int parent = in.parent[static_cast<std::size_t>(v)];
            if (preorder_position[static_cast<std::size_t>(parent)] >=
                preorder_position[static_cast<std::size_t>(v)])
                throw std::invalid_argument("preorder places a child before its parent");
        }
    }

    for (int v = 0; v < in.m; ++v) {
        const auto& tuples = in.rel[static_cast<std::size_t>(v)].tuples;
        if (tuples.size() >
            static_cast<std::size_t>(std::numeric_limits<int>::max()))
            throw std::length_error("relation exceeds int tuple-index range");
        std::unordered_set<int> ids;
        ids.reserve(tuples.size());
        for (const Tuple& tuple : tuples) {
            if (!std::isfinite(tuple.weight) || tuple.weight < 0.0 ||
                tuple.weight > 1.0)
                throw std::invalid_argument(
                    "tuple weight must be finite and lie in [0,1]");
            if (tuple.edge_key.size() != m)
                throw std::invalid_argument("tuple edge-key vector size mismatch m");
            if (!ids.insert(tuple.id).second)
                throw std::invalid_argument("tuple ids must be unique per relation");
        }
    }
}

inline double uniform_open(std::mt19937_64& rng) {
    
    const std::uint64_t x = rng() >> 12;
    return (static_cast<double>(x) + 0.5) *
           (1.0 / 4503599627370496.0);
}

inline bool coin(std::mt19937_64& rng, double p) {
    if (p <= 0.0) return false;
    if (p >= 1.0) return true;
    return uniform_open(rng) < p;
}

inline double log_survival_ratio(double p) {
    return p > 0.0 ? -std::log1p(-p) / p : 1.0;
}

struct AliasTable {
    std::vector<int> item;
    std::vector<double> prob;
    std::vector<std::size_t> alias;

    void build(const std::vector<int>& candidates,
               const std::vector<double>& weights) {
        if (candidates.size() != weights.size())
            throw std::invalid_argument("alias candidates/weights size mismatch");
        item.clear();
        prob.clear();
        alias.clear();
        double total = 0.0;
        for (std::size_t i = 0; i < candidates.size(); ++i) {
            if (!std::isfinite(weights[i]) || weights[i] < 0.0)
                throw std::invalid_argument("alias weight must be finite and nonnegative");
            if (weights[i] > 0.0) {
                item.push_back(candidates[i]);
                total += weights[i];
                if (!std::isfinite(total))
                    throw std::overflow_error("alias total weight is not finite");
            }
        }
        if (item.empty()) return;
        if (!(total > 0.0) || !std::isfinite(total))
            throw std::overflow_error("positive alias total is not finite");
        const std::size_t n = item.size();
        prob.assign(n, 0.0);
        alias.assign(n, 0);
        std::vector<double> scaled(n);
        std::vector<std::size_t> small, large;
        small.reserve(n);
        large.reserve(n);
        std::size_t k = 0;
        for (std::size_t i = 0; i < candidates.size(); ++i) {
            if (weights[i] <= 0.0) continue;
            scaled[k] = weights[i] * static_cast<double>(n) / total;
            if (!std::isfinite(scaled[k]))
                throw std::overflow_error("alias scaled weight is not finite");
            (scaled[k] < 1.0 ? small : large).push_back(k);
            ++k;
        }
        while (!small.empty() && !large.empty()) {
            const std::size_t s = small.back();
            small.pop_back();
            const std::size_t l = large.back();
            large.pop_back();
            prob[s] = scaled[s];
            alias[s] = l;
            scaled[l] = (scaled[l] + scaled[s]) - 1.0;
            (scaled[l] < 1.0 ? small : large).push_back(l);
        }
        for (std::size_t i : large) {
            prob[i] = 1.0;
            alias[i] = i;
        }
        for (std::size_t i : small) {
            prob[i] = 1.0;
            alias[i] = i;
        }
    }

    bool empty() const { return item.empty(); }

    int sample(std::mt19937_64& rng) const {
        if (item.empty())
            throw std::runtime_error("sampling an empty alias table");
        if (item.size() == 1) return item.front();
        std::uniform_int_distribution<std::size_t> column(0, item.size() - 1);
        const std::size_t i = column(rng);
        const std::size_t j = uniform_open(rng) < prob[i] ? i : alias[i];
        return item[j];
    }
};

struct WeightedGroup {
    Key key = 0;
    std::vector<int> tuples;
    double Z = 0.0;
    double M = 0.0;
    AliasTable alias;
    std::vector<int> heap;
};

struct WeightedNode {
    std::unordered_map<Key, std::size_t> lookup;
    std::vector<WeightedGroup> groups;
    std::vector<double> zscore;
    std::vector<double> mscore;
};

struct HeavyStats {
    std::uint64_t heap_examined = 0;
    std::uint64_t passing_prefixes = 0;
    std::uint64_t outputs = 0;
};

struct DrawnOutput {
    Output ids;
    double p = 0.0;
};

class WeightedJoinIndex {
public:
    explicit WeightedJoinIndex(const Instance& instance)
        : in_(instance) {
        validate_instance(instance);
        node_.resize(static_cast<std::size_t>(instance.m));
        build();
    }
    WeightedJoinIndex(Instance&&) = delete;

    double mu() const { return mu_; }
    double max_probability() const { return max_probability_; }
    std::size_t relation_count() const {
        return static_cast<std::size_t>(in_.m);
    }
    double poisson_factor() const {
        const double exact_model_factor =
            log_survival_ratio(std::min(max_probability_, 0.5));
        const double slack = 1.0 + 64.0 * static_cast<double>(in_.m) *
            std::numeric_limits<double>::epsilon();
        const double factor = std::nextafter(
            exact_model_factor * slack,
            std::numeric_limits<double>::infinity());
        if (!(factor > 0.0) || !std::isfinite(factor))
            throw std::overflow_error("Poisson factor is not finite and positive");
        return factor;
    }

    std::size_t approximate_bytes() const {
        std::size_t n = sizeof(*this);
        for (const auto& nd : node_) {
            n += nd.zscore.capacity() * sizeof(double);
            n += nd.mscore.capacity() * sizeof(double);
            n += nd.lookup.size() * (sizeof(Key) + sizeof(std::size_t) + 24);
            n += nd.groups.capacity() * sizeof(WeightedGroup);
            for (const auto& group : nd.groups) {
                n += group.tuples.capacity() * sizeof(int);
                n += group.heap.capacity() * sizeof(int);
                n += group.alias.item.capacity() * sizeof(int);
                n += group.alias.prob.capacity() * sizeof(double);
                n += group.alias.alias.capacity() * sizeof(std::size_t);
            }
        }
        return n;
    }

    double draw_into(std::mt19937_64& rng, Output& ids,
                     std::vector<int>& chosen) const {
        // Proportional draws cover the complete join.
        if (!(mu_ > 0.0))
            throw std::runtime_error("draw on zero-mass join");
        if (ids.size() != static_cast<std::size_t>(in_.m))
            ids.resize(static_cast<std::size_t>(in_.m));
        if (chosen.size() != static_cast<std::size_t>(in_.m))
            chosen.resize(static_cast<std::size_t>(in_.m));
        double output_probability = 1.0;
        for (int v : in_.preorder) {
            const WeightedGroup* group = nullptr;
            if (in_.parent[static_cast<std::size_t>(v)] < 0) {
                group = find_group(v, ROOT_KEY);
            } else {
                const int parent = in_.parent[static_cast<std::size_t>(v)];
                const Tuple& parent_tuple =
                    in_.rel[static_cast<std::size_t>(parent)].tuples[
                        static_cast<std::size_t>(chosen[static_cast<std::size_t>(parent)])];
                group = find_group(v,
                    parent_tuple.edge_key[static_cast<std::size_t>(v)]);
            }
            if (!group || group->alias.empty())
                throw std::runtime_error("positive draw reached empty group");
            const int tuple_index = group->alias.sample(rng);
            chosen[static_cast<std::size_t>(v)] = tuple_index;
            const Tuple& tuple = in_.rel[static_cast<std::size_t>(v)].tuples[
                static_cast<std::size_t>(tuple_index)];
            ids[static_cast<std::size_t>(v)] = tuple.id;
            output_probability *= tuple.weight;
        }
        if (!std::isfinite(output_probability))
            throw std::overflow_error("drawn output probability is not finite");
        return output_probability;
    }

    DrawnOutput draw(std::mt19937_64& rng) const {
        DrawnOutput output;
        std::vector<int> chosen;
        output.p = draw_into(rng, output.ids, chosen);
        return output;
    }

    template<class Callback>
    void enumerate_heavy(Callback&& callback, HeavyStats& stats) const {
        const WeightedGroup* root =
            find_group(in_.preorder.front(), ROOT_KEY);

        const double m = static_cast<double>(in_.m);
        const double margin = 128.0 * m * m *
            std::numeric_limits<double>::epsilon();
        if (!(margin < 0.01))
            throw std::runtime_error("heavy guard precision range exceeded");
        // Admit boundary candidates during search, then classify their complete products.
        const double search_cutoff = 0.5 / (1.0 + margin);
        if (!root || !(root->M > search_cutoff)) return;
        Output ids(static_cast<std::size_t>(in_.m), -1);
        std::vector<int> chosen(static_cast<std::size_t>(in_.m), -1);
        heavy_dfs(0, root->M, 1.0, search_cutoff, ids, chosen, stats,
                  callback);
    }

private:
    const Instance& in_;
    std::vector<WeightedNode> node_;
    double mu_ = 0.0;
    double max_probability_ = 0.0;

    WeightedGroup* find_group_mut(int v, Key key) {
        auto& node = node_[static_cast<std::size_t>(v)];
        auto it = node.lookup.find(key);
        return it == node.lookup.end() ? nullptr : &node.groups[it->second];
    }

    const WeightedGroup* find_group(int v, Key key) const {
        const auto& node = node_[static_cast<std::size_t>(v)];
        auto it = node.lookup.find(key);
        return it == node.lookup.end() ? nullptr : &node.groups[it->second];
    }

    void build() {
        for (int v = 0; v < in_.m; ++v) {
            auto& node = node_[static_cast<std::size_t>(v)];
            const auto& tuples = in_.rel[static_cast<std::size_t>(v)].tuples;
            node.zscore.assign(tuples.size(), 0.0);
            node.mscore.assign(tuples.size(), 0.0);
            for (std::size_t i = 0; i < tuples.size(); ++i) {
                const Tuple& tuple = tuples[i];
                const int parent = in_.parent[static_cast<std::size_t>(v)];
                const Key key = parent < 0 ? ROOT_KEY
                    : tuple.edge_key[static_cast<std::size_t>(parent)];
                auto [it, inserted] =
                    node.lookup.emplace(key, node.groups.size());
                if (inserted) {
                    node.groups.push_back(WeightedGroup{});
                    node.groups.back().key = key;
                }
                node.groups[it->second].tuples.push_back(static_cast<int>(i));
            }
        }

        for (auto it = in_.preorder.rbegin(); it != in_.preorder.rend(); ++it) {
            const int v = *it;
            auto& node = node_[static_cast<std::size_t>(v)];
            const auto& tuples = in_.rel[static_cast<std::size_t>(v)].tuples;
            for (std::size_t i = 0; i < tuples.size(); ++i) {
                const Tuple& tuple = tuples[i];
                double mass = tuple.weight;
                double maximum = tuple.weight;
                for (int child : in_.children[static_cast<std::size_t>(v)]) {
                    const WeightedGroup* child_group = find_group(
                        child, tuple.edge_key[static_cast<std::size_t>(child)]);
                    if (!child_group) {
                        mass = 0.0;
                        maximum = 0.0;
                        break;
                    }
                    mass *= child_group->Z;
                    maximum *= child_group->M;
                    if (!std::isfinite(mass))
                        throw std::overflow_error(
                            "weighted Yannakakis tuple mass is not finite");
                    if (!std::isfinite(maximum))
                        throw std::overflow_error(
                            "weighted Yannakakis max product is not finite");
                }
                node.zscore[i] = mass;
                node.mscore[i] = maximum;
            }
            for (auto& group : node.groups) {
                std::vector<double> weights;
                weights.reserve(group.tuples.size());
                group.Z = 0.0;
                group.M = 0.0;
                for (int tuple_index : group.tuples) {
                    const double mass =
                        node.zscore[static_cast<std::size_t>(tuple_index)];
                    const double maximum =
                        node.mscore[static_cast<std::size_t>(tuple_index)];
                    group.Z += mass;
                    if (!std::isfinite(group.Z))
                        throw std::overflow_error(
                            "weighted Yannakakis group mass is not finite");
                    group.M = std::max(group.M, maximum);
                    weights.push_back(mass);
                }
                group.alias.build(group.tuples, weights);
                group.heap = group.tuples;
                std::make_heap(
                    group.heap.begin(), group.heap.end(),
                    [&](int a, int b) {
                        return node.mscore[static_cast<std::size_t>(a)] <
                               node.mscore[static_cast<std::size_t>(b)];
                    });
            }
        }
        const WeightedGroup* root =
            find_group(in_.preorder.front(), ROOT_KEY);
        mu_ = root ? root->Z : 0.0;
        max_probability_ = root ? root->M : 0.0;
        if (!std::isfinite(mu_) || mu_ < 0.0)
            throw std::overflow_error("root join mass is not finite");
        if (!std::isfinite(max_probability_) || max_probability_ < 0.0 ||
            max_probability_ > 1.0)
            throw std::overflow_error("root max probability is invalid");
    }

    template<class Callback>
    void heavy_dfs(std::size_t depth, double bound,
                   double prefix_probability, double search_cutoff,
                   Output& ids, std::vector<int>& chosen,
                   HeavyStats& stats, Callback& callback) const {
        const int v = in_.preorder[depth];
        const WeightedGroup* group = nullptr;
        if (in_.parent[static_cast<std::size_t>(v)] < 0) {
            group = find_group(v, ROOT_KEY);
        } else {
            const int parent = in_.parent[static_cast<std::size_t>(v)];
            const Tuple& parent_tuple =
                in_.rel[static_cast<std::size_t>(parent)].tuples[
                    static_cast<std::size_t>(chosen[static_cast<std::size_t>(parent)])];
            group = find_group(v,
                parent_tuple.edge_key[static_cast<std::size_t>(v)]);
        }
        if (!group || !(group->M > 0.0)) return;
        const double threshold = group->M * search_cutoff / bound;
        std::vector<std::size_t> stack;
        if (!group->heap.empty()) stack.push_back(0);
        while (!stack.empty()) {
            const std::size_t heap_position = stack.back();
            stack.pop_back();
            ++stats.heap_examined;
            const int tuple_index = group->heap[heap_position];
            const double score = node_[static_cast<std::size_t>(v)].mscore[
                static_cast<std::size_t>(tuple_index)];
            if (!(score > threshold)) continue;
            ++stats.passing_prefixes;
            const double next_bound = bound * score / group->M;
            const Tuple& tuple = in_.rel[static_cast<std::size_t>(v)].tuples[
                static_cast<std::size_t>(tuple_index)];
            const double next_probability = prefix_probability * tuple.weight;
            chosen[static_cast<std::size_t>(v)] = tuple_index;
            ids[static_cast<std::size_t>(v)] = tuple.id;
            if (depth + 1 == in_.preorder.size()) {
                if (next_probability > 0.5) {
                    ++stats.outputs;
                    callback(ids, next_probability);
                }
            } else {
                heavy_dfs(depth + 1, next_bound, next_probability,
                          search_cutoff, ids, chosen, stats, callback);
            }
            const std::size_t right = heap_position * 2 + 2;
            const std::size_t left = heap_position * 2 + 1;
            if (right < group->heap.size()) stack.push_back(right);
            if (left < group->heap.size()) stack.push_back(left);
        }
    }
};

struct QueryStats {
    std::uint64_t proposals = 0;
    std::uint64_t clocks = 0;
    std::uint64_t accepted_occurrences = 0;
    std::uint64_t heavy_outputs = 0;
    HeavyStats heavy;
};

inline std::unordered_set<Output, VecHash>
sample_subset(const WeightedJoinIndex& index, std::mt19937_64& rng,
              QueryStats& stats,
              bool validate_heavy_membership = false) {
    std::unordered_set<Output, VecHash> result;
    const double mu = index.mu();
    if (!(mu > 0.0)) return result;
    if (!std::isfinite(mu))
        throw std::overflow_error("positive join mass is not finite");
    const double factor = index.poisson_factor();
    const double rate = factor * mu;
    if (!(rate > 0.0) || !std::isfinite(rate))
        throw std::overflow_error("Poisson clock rate is not finite and positive");

    std::unordered_set<Output, VecHash> heavy_ids;
    index.enumerate_heavy(
        [&](const Output& ids, double probability) {
            if (validate_heavy_membership) heavy_ids.insert(ids);
            if (coin(rng, probability)) result.insert(ids);
        },
        stats.heavy);
    stats.heavy_outputs = stats.heavy.outputs;

    Output drawn_ids(index.relation_count(), -1);
    std::vector<int> chosen(index.relation_count(), -1);
    double time = 0.0;
    while (true) {
        ++stats.clocks;
        time += -std::log(uniform_open(rng)) / rate;
        if (time > 1.0) break;
        ++stats.proposals;
        const double probability = index.draw_into(rng, drawn_ids, chosen);
        if (probability > 0.5) {
            if (validate_heavy_membership &&
                heavy_ids.find(drawn_ids) == heavy_ids.end())
                throw std::runtime_error(
                    "weighted draw found heavy output missed by enumerator");
            continue;
        }
        if (!(probability > 0.0) || !std::isfinite(probability))
            throw std::runtime_error("invalid light proposal probability");
        const double accept = log_survival_ratio(probability) / factor;
        if (!(accept >= 0.0 && accept <= 1.0) || !std::isfinite(accept))
            throw std::runtime_error("floating thinning bound violated");
        // Accept each proposal independently before retaining one copy of its vector.
        if (coin(rng, accept)) {
            ++stats.accepted_occurrences;
            result.insert(drawn_ids);
        }
    }
    return result;
}
