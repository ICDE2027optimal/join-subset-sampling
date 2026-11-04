// Validation helpers for the join-sampling implementation.
#pragma once

#include "core.hpp"

#include <functional>

inline std::vector<DrawnOutput> oracle_join(const Instance& in) {
    std::vector<DrawnOutput> output;
    Output ids(static_cast<std::size_t>(in.m), -1);
    std::vector<int> chosen(static_cast<std::size_t>(in.m), -1);
    std::function<void(std::size_t, double)> recurse =
        [&](std::size_t depth, double probability) {
            if (depth == in.preorder.size()) {
                output.push_back(DrawnOutput{ids, probability});
                return;
            }
            const int v = in.preorder[depth];
            const auto& tuples = in.rel[static_cast<std::size_t>(v)].tuples;
            for (std::size_t ti = 0; ti < tuples.size(); ++ti) {
                const Tuple& tuple = tuples[ti];
                if (in.parent[static_cast<std::size_t>(v)] >= 0) {
                    const int parent = in.parent[static_cast<std::size_t>(v)];
                    const Tuple& parent_tuple =
                        in.rel[static_cast<std::size_t>(parent)].tuples[
                            static_cast<std::size_t>(
                                chosen[static_cast<std::size_t>(parent)])];
                    if (tuple.edge_key[static_cast<std::size_t>(parent)] !=
                        parent_tuple.edge_key[static_cast<std::size_t>(v)])
                        continue;
                }
                chosen[static_cast<std::size_t>(v)] = static_cast<int>(ti);
                ids[static_cast<std::size_t>(v)] = tuple.id;
                recurse(depth + 1, probability * tuple.weight);
            }
        };
    recurse(0, 1.0);
    return output;
}

inline std::vector<int>
make_preorder(const std::vector<std::vector<int>>& children) {
    std::vector<int> order;
    std::function<void(int)> visit = [&](int v) {
        order.push_back(v);
        for (int child : children[static_cast<std::size_t>(v)]) visit(child);
    };
    visit(0);
    return order;
}

inline Instance manual_chain3() {
    Instance in;
    in.m = 3;
    in.shape = "chain";
    in.weight_mode = "edgecases";
    in.parent = {-1, 0, 1};
    in.children = {{1}, {2}, {}};
    in.preorder = make_preorder(in.children);
    in.rel.resize(3);
    int id = 0;
    auto add = [&](int v, double weight, Key parent_key, Key child_key) {
        Tuple tuple;
        tuple.id = id++;
        tuple.weight = weight;
        tuple.edge_key.assign(3, -1);
        if (in.parent[static_cast<std::size_t>(v)] >= 0)
            tuple.edge_key[static_cast<std::size_t>(
                in.parent[static_cast<std::size_t>(v)])] = parent_key;
        if (!in.children[static_cast<std::size_t>(v)].empty())
            tuple.edge_key[static_cast<std::size_t>(
                in.children[static_cast<std::size_t>(v)].front())] = child_key;
        in.rel[static_cast<std::size_t>(v)].tuples.push_back(std::move(tuple));
    };
    add(0, 1.0, -1, 10);
    add(0, 0.0, -1, 20);
    add(0, 0.9, -1, 9999);
    add(1, 1.0, 10, 30);
    add(1, 0.8, 10, 31);
    add(1, 1.0, 20, 32);
    add(1, 0.7, 7777, 33);
    add(2, 1.0, 30, -1);
    add(2, 0.5, 30, -1);
    add(2, 0.9, 31, -1);
    add(2, 0.25, 31, -1);
    add(2, 1.0, 32, -1);
    add(2, 1.0, 8888, -1);
    return in;
}

inline Instance zero_mass_case() {
    Instance in = manual_chain3();
    for (auto& relation : in.rel)
        for (auto& tuple : relation.tuples) tuple.weight = 0.0;
    in.weight_mode = "all_zero";
    return in;
}

inline double skew_raw(std::mt19937_64& rng) {
    const double u = uniform_open(rng);
    if (u < 0.25) return 1.0;
    if (u < 0.60) return 0.8;
    if (u < 0.85) return 0.55;
    return 0.25;
}

inline Instance generate_instance(const std::string& shape, int m, int roots,
                                  int fanout, const std::string& mode,
                                  double scale, std::uint64_t seed,
                                  bool dangling = true) {
    Instance in;
    in.m = m;
    in.shape = shape;
    in.weight_mode = mode;
    in.parent.assign(static_cast<std::size_t>(m), -1);
    in.children.assign(static_cast<std::size_t>(m), {});
    if (shape == "chain") {
        for (int v = 1; v < m; ++v) {
            in.parent[static_cast<std::size_t>(v)] = v - 1;
            in.children[static_cast<std::size_t>(v - 1)].push_back(v);
        }
    } else if (shape == "branch") {
        for (int v = 1; v < m; ++v) {
            in.parent[static_cast<std::size_t>(v)] = 0;
            in.children[0].push_back(v);
        }
    } else {
        throw std::invalid_argument("unknown shape");
    }
    in.preorder = make_preorder(in.children);
    in.rel.resize(static_cast<std::size_t>(m));
    std::mt19937_64 rng(seed);
    int next_id = 0;
    Key next_key = 100;
    auto weight = [&]() {
        const double raw = mode == "uniform" ? 1.0 : skew_raw(rng);
        return std::min(1.0, raw * scale);
    };
    std::function<void(int, int)> spawn = [&](int v, int ti) {
        for (int child : in.children[static_cast<std::size_t>(v)]) {
            const Key key = next_key++;
            in.rel[static_cast<std::size_t>(v)].tuples[
                static_cast<std::size_t>(ti)].edge_key[
                    static_cast<std::size_t>(child)] = key;
            for (int j = 0; j < fanout; ++j) {
                Tuple tuple;
                tuple.id = next_id++;
                tuple.weight = weight();
                tuple.edge_key.assign(static_cast<std::size_t>(m), -1);
                tuple.edge_key[static_cast<std::size_t>(v)] = key;
                in.rel[static_cast<std::size_t>(child)].tuples.push_back(
                    std::move(tuple));
                spawn(child, static_cast<int>(
                    in.rel[static_cast<std::size_t>(child)].tuples.size() - 1));
            }
        }
    };
    for (int i = 0; i < roots; ++i) {
        Tuple tuple;
        tuple.id = next_id++;
        tuple.weight = weight();
        tuple.edge_key.assign(static_cast<std::size_t>(m), -1);
        in.rel[0].tuples.push_back(std::move(tuple));
        spawn(0, static_cast<int>(in.rel[0].tuples.size() - 1));
    }
    if (dangling) {
        for (int v = 1; v < m; ++v) {
            Tuple tuple;
            tuple.id = next_id++;
            tuple.weight = weight();
            tuple.edge_key.assign(static_cast<std::size_t>(m), -1);
            tuple.edge_key[static_cast<std::size_t>(
                in.parent[static_cast<std::size_t>(v)])] = next_key++;
            in.rel[static_cast<std::size_t>(v)].tuples.push_back(
                std::move(tuple));
        }
        if (!in.children[0].empty()) {
            Tuple tuple;
            tuple.id = next_id++;
            tuple.weight = weight();
            tuple.edge_key.assign(static_cast<std::size_t>(m), -1);
            for (int child : in.children[0])
                tuple.edge_key[static_cast<std::size_t>(child)] = next_key++;
            in.rel[0].tuples.push_back(std::move(tuple));
        }
    }
    return in;
}

inline bool output_less(const DrawnOutput& a, const DrawnOutput& b) {
    return a.ids < b.ids;
}

struct CountGroup {
    Key key = 0;
    std::vector<int> tuples;
    std::vector<std::uint64_t> cumulative;
    std::uint64_t total = 0;
};

struct CountNode {
    std::unordered_map<Key, std::size_t> lookup;
    std::vector<CountGroup> groups;
    std::vector<std::uint64_t> tuple_count;
};

inline std::uint64_t saturated_add(std::uint64_t a, std::uint64_t b) {
    if (std::numeric_limits<std::uint64_t>::max() - a < b)
        return std::numeric_limits<std::uint64_t>::max();
    return a + b;
}

inline std::uint64_t saturated_multiply(std::uint64_t a, std::uint64_t b) {
    if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a)
        return std::numeric_limits<std::uint64_t>::max();
    return a * b;
}

class RankIndex {
public:
    explicit RankIndex(const Instance& in)
        : in_(in), node_(static_cast<std::size_t>(in.m)) {
        validate_instance(in);
        build();
    }

    std::uint64_t join_size() const { return join_size_; }

    Output access(std::uint64_t rank) const {
        if (rank >= join_size_) throw std::out_of_range("rank access");
        Output output(static_cast<std::size_t>(in_.m), -1);
        access_group(in_.preorder.front(), ROOT_KEY, rank, output);
        return output;
    }

    template<class Callback>
    void enumerate_weighted(Callback&& callback) const {
        if (join_size_ == 0) return;
        Output ids(static_cast<std::size_t>(in_.m), -1);
        std::vector<int> chosen(static_cast<std::size_t>(in_.m), -1);
        enumerate_group(0, ids, chosen, 1.0, callback);
    }

private:
    const Instance& in_;
    std::vector<CountNode> node_;
    std::uint64_t join_size_ = 0;

    const CountGroup* find_group(int v, Key key) const {
        const auto& node = node_[static_cast<std::size_t>(v)];
        auto it = node.lookup.find(key);
        return it == node.lookup.end() ? nullptr : &node.groups[it->second];
    }

    void build() {
        for (int v = 0; v < in_.m; ++v) {
            auto& node = node_[static_cast<std::size_t>(v)];
            const auto& tuples = in_.rel[static_cast<std::size_t>(v)].tuples;
            node.tuple_count.assign(tuples.size(), 0);
            for (std::size_t i = 0; i < tuples.size(); ++i) {
                const Tuple& tuple = tuples[i];
                const int parent = in_.parent[static_cast<std::size_t>(v)];
                const Key key = parent < 0 ? ROOT_KEY
                    : tuple.edge_key[static_cast<std::size_t>(parent)];
                auto [it, inserted] =
                    node.lookup.emplace(key, node.groups.size());
                if (inserted) {
                    node.groups.push_back(CountGroup{});
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
                std::uint64_t count = 1;
                for (int child : in_.children[static_cast<std::size_t>(v)]) {
                    const CountGroup* child_group = find_group(
                        child, tuple.edge_key[static_cast<std::size_t>(child)]);
                    if (!child_group) {
                        count = 0;
                        break;
                    }
                    count = saturated_multiply(count, child_group->total);
                }
                node.tuple_count[i] = count;
            }
            for (auto& group : node.groups) {
                group.cumulative.clear();
                group.cumulative.reserve(group.tuples.size());
                group.total = 0;
                for (int tuple_index : group.tuples) {
                    group.total = saturated_add(
                        group.total,
                        node.tuple_count[static_cast<std::size_t>(tuple_index)]);
                    group.cumulative.push_back(group.total);
                }
            }
        }
        const CountGroup* root =
            find_group(in_.preorder.front(), ROOT_KEY);
        join_size_ = root ? root->total : 0;
    }

    void access_group(int v, Key key, std::uint64_t rank,
                      Output& output) const {
        const CountGroup* group = find_group(v, key);
        if (!group || rank >= group->total)
            throw std::runtime_error("invalid child rank");
        auto it = std::upper_bound(
            group->cumulative.begin(), group->cumulative.end(), rank);
        const std::size_t position =
            static_cast<std::size_t>(it - group->cumulative.begin());
        const std::uint64_t before =
            position == 0 ? 0 : group->cumulative[position - 1];
        std::uint64_t local = rank - before;
        const int tuple_index = group->tuples[position];
        const Tuple& tuple = in_.rel[static_cast<std::size_t>(v)].tuples[
            static_cast<std::size_t>(tuple_index)];
        output[static_cast<std::size_t>(v)] = tuple.id;
        const auto& children = in_.children[static_cast<std::size_t>(v)];
        for (std::size_t j = 0; j < children.size(); ++j) {
            std::uint64_t suffix = 1;
            for (std::size_t k = j + 1; k < children.size(); ++k) {
                const CountGroup* later = find_group(
                    children[k],
                    tuple.edge_key[static_cast<std::size_t>(children[k])]);
                suffix = saturated_multiply(suffix, later ? later->total : 0);
            }
            const int child = children[j];
            const CountGroup* child_group = find_group(
                child, tuple.edge_key[static_cast<std::size_t>(child)]);
            if (!child_group || child_group->total == 0 || suffix == 0)
                throw std::runtime_error("zero child in positive rank");
            const std::uint64_t child_rank = local / suffix;
            local %= suffix;
            access_group(child,
                         tuple.edge_key[static_cast<std::size_t>(child)],
                         child_rank, output);
        }
    }

    template<class Callback>
    void enumerate_group(std::size_t depth, Output& ids,
                         std::vector<int>& chosen, double probability,
                         Callback& callback) const {
        const int v = in_.preorder[depth];
        Key key = ROOT_KEY;
        if (in_.parent[static_cast<std::size_t>(v)] >= 0) {
            const int parent = in_.parent[static_cast<std::size_t>(v)];
            key = in_.rel[static_cast<std::size_t>(parent)].tuples[
                static_cast<std::size_t>(chosen[static_cast<std::size_t>(parent)])]
                      .edge_key[static_cast<std::size_t>(v)];
        }
        const CountGroup* group = find_group(v, key);
        if (!group) return;
        for (int tuple_index : group->tuples) {
            if (node_[static_cast<std::size_t>(v)].tuple_count[
                    static_cast<std::size_t>(tuple_index)] == 0)
                continue;
            const Tuple& tuple = in_.rel[static_cast<std::size_t>(v)].tuples[
                static_cast<std::size_t>(tuple_index)];
            chosen[static_cast<std::size_t>(v)] = tuple_index;
            ids[static_cast<std::size_t>(v)] = tuple.id;
            const double next_probability = probability * tuple.weight;
            if (depth + 1 == in_.preorder.size())
                callback(ids, next_probability);
            else
                enumerate_group(depth + 1, ids, chosen, next_probability,
                                callback);
        }
    }
};
