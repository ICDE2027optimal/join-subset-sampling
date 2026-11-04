// Uniform rank sampling and explicit Bernoulli scanning baselines.
#pragma once

#include "core.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rank_scan_controls {

using Count = std::uint64_t;

inline Count checked_add(Count a, Count b, const char* where) {
    if (std::numeric_limits<Count>::max() - a < b)
        throw std::overflow_error(std::string("uint64 count addition overflow at ") + where);
    return a + b;
}

inline Count checked_multiply(Count a, Count b, const char* where) {
    if (a != 0 && b > std::numeric_limits<Count>::max() / a)
        throw std::overflow_error(std::string("uint64 count multiplication overflow at ") + where);
    return a * b;
}

inline std::size_t checked_size_add(std::size_t a, std::size_t b,
                                    const char* where) {
    if (std::numeric_limits<std::size_t>::max() - a < b)
        throw std::overflow_error(std::string("size accounting overflow at ") + where);
    return a + b;
}

inline std::size_t checked_size_multiply(std::size_t a, std::size_t b,
                                         const char* where) {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a)
        throw std::overflow_error(std::string("size accounting overflow at ") + where);
    return a * b;
}

struct RankBuildStats {
    std::size_t input_tuples = 0;
    std::size_t groups = 0;
    std::size_t positive_tuple_entries = 0;
    std::size_t discarded_zero_tuple_entries = 0;
    std::size_t prefix_cells = 0;
    std::size_t suffix_cells = 0;
    Count join_size = 0;
    std::size_t approximate_bytes = 0;
};

struct EnumerationStats {
    Count tuple_extensions = 0;
    Count outputs = 0;
};

namespace detail {

struct CountGroup {
    Key key = 0;
    
    std::vector<int> tuples;
    std::vector<Count> cumulative;
    Count total = 0;
};

struct CountNode {
    std::unordered_map<Key, std::size_t> lookup;
    std::vector<CountGroup> groups;
    std::vector<Count> tuple_total;
    
    std::vector<std::vector<Count>> suffix;
};

class CheckedRankIndex {
public:
    explicit CheckedRankIndex(const Instance& instance)
        : in_(instance), node_(static_cast<std::size_t>(instance.m)) {
        validate_instance(in_);
        build();
        finalize_stats();
    }

    CheckedRankIndex(const CheckedRankIndex&) = delete;
    CheckedRankIndex& operator=(const CheckedRankIndex&) = delete;

    Count join_size() const { return join_size_; }
    const RankBuildStats& build_stats() const { return stats_; }
    std::size_t approximate_bytes() const { return stats_.approximate_bytes; }

    Output access(Count rank) const {
        if (rank >= join_size_) throw std::out_of_range("join rank access");
        Output output(static_cast<std::size_t>(in_.m), -1);
        access_group(in_.preorder.front(), ROOT_KEY, rank, output);
        return output;
    }

    template<class Callback>
    void enumerate_weighted(Callback&& callback,
                            EnumerationStats* enumeration = nullptr) const {
        if (join_size_ == 0) return;
        Output ids(static_cast<std::size_t>(in_.m), -1);
        std::vector<int> chosen(static_cast<std::size_t>(in_.m), -1);
        auto&& callback_ref = callback;
        enumerate_group(0, ids, chosen, 1.0, callback_ref, enumeration);
    }

private:
    const Instance& in_;
    std::vector<CountNode> node_;
    Count join_size_ = 0;
    RankBuildStats stats_;

    const CountGroup* find_group(int relation, Key key) const {
        const auto& node = node_[static_cast<std::size_t>(relation)];
        auto it = node.lookup.find(key);
        return it == node.lookup.end() ? nullptr : &node.groups[it->second];
    }

    void build() {
        for (int relation = 0; relation < in_.m; ++relation) {
            auto& node = node_[static_cast<std::size_t>(relation)];
            const auto& tuples = in_.rel[static_cast<std::size_t>(relation)].tuples;
            stats_.input_tuples = checked_size_add(
                stats_.input_tuples, tuples.size(), "rank input tuples");
            node.tuple_total.assign(tuples.size(), 0);
            node.suffix.resize(tuples.size());
            for (std::size_t tuple_index = 0; tuple_index < tuples.size();
                 ++tuple_index) {
                const Tuple& tuple = tuples[tuple_index];
                const int parent = in_.parent[static_cast<std::size_t>(relation)];
                const Key key = parent < 0
                    ? ROOT_KEY
                    : tuple.edge_key[static_cast<std::size_t>(parent)];
                auto [position, inserted] =
                    node.lookup.emplace(key, node.groups.size());
                if (inserted) {
                    node.groups.push_back(CountGroup{});
                    node.groups.back().key = key;
                }
                node.groups[position->second].tuples.push_back(
                    static_cast<int>(tuple_index));
            }
        }

        for (auto order = in_.preorder.rbegin(); order != in_.preorder.rend();
             ++order) {
            const int relation = *order;
            auto& node = node_[static_cast<std::size_t>(relation)];
            const auto& tuples = in_.rel[static_cast<std::size_t>(relation)].tuples;
            const auto& children = in_.children[static_cast<std::size_t>(relation)];
            for (std::size_t tuple_index = 0; tuple_index < tuples.size();
                 ++tuple_index) {
                const Tuple& tuple = tuples[tuple_index];
                auto& suffix = node.suffix[tuple_index];
                suffix.assign(children.size() + 1, 1);
                for (std::size_t child_position = children.size();
                     child_position-- > 0;) {
                    const int child = children[child_position];
                    const CountGroup* child_group = find_group(
                        child, tuple.edge_key[static_cast<std::size_t>(child)]);
                    const Count child_total = child_group ? child_group->total : 0;
                    suffix[child_position] = checked_multiply(
                        child_total, suffix[child_position + 1],
                        "rank child suffix");
                }
                node.tuple_total[tuple_index] = suffix[0];
            }

            for (auto& group : node.groups) {
                std::vector<int> positive;
                std::vector<Count> cumulative;
                positive.reserve(group.tuples.size());
                cumulative.reserve(group.tuples.size());
                Count total = 0;
                for (int tuple_index : group.tuples) {
                    const Count completions = node.tuple_total[
                        static_cast<std::size_t>(tuple_index)];
                    if (completions == 0) {
                        stats_.discarded_zero_tuple_entries = checked_size_add(
                            stats_.discarded_zero_tuple_entries, 1,
                            "discarded zero-completion tuples");
                        continue;
                    }
                    total = checked_add(total, completions,
                                        "rank group prefix");
                    positive.push_back(tuple_index);
                    cumulative.push_back(total);
                    stats_.positive_tuple_entries = checked_size_add(
                        stats_.positive_tuple_entries, 1,
                        "positive-completion tuples");
                }
                group.tuples.swap(positive);
                group.cumulative.swap(cumulative);
                group.total = total;
            }

            for (std::size_t tuple_index = 0; tuple_index < tuples.size();
                 ++tuple_index) {
                if (node.tuple_total[tuple_index] == 0) {
                    std::vector<Count> empty;
                    node.suffix[tuple_index].swap(empty);
                }
            }
        }

        const CountGroup* root = find_group(in_.preorder.front(), ROOT_KEY);
        join_size_ = root ? root->total : 0;
        stats_.join_size = join_size_;
    }

    void finalize_stats() {
        std::size_t bytes = sizeof(*this);
        for (const auto& node : node_) {
            stats_.groups = checked_size_add(
                stats_.groups, node.groups.size(), "rank group count");
            bytes = checked_size_add(bytes,
                checked_size_multiply(node.lookup.size(),
                    sizeof(Key) + sizeof(std::size_t) + 24,
                    "rank lookup bytes"),
                "rank bytes");
            bytes = checked_size_add(bytes,
                checked_size_multiply(node.groups.capacity(), sizeof(CountGroup),
                    "rank group bytes"),
                "rank bytes");
            bytes = checked_size_add(bytes,
                checked_size_multiply(node.tuple_total.capacity(), sizeof(Count),
                    "rank tuple-count bytes"),
                "rank bytes");
            bytes = checked_size_add(bytes,
                checked_size_multiply(node.suffix.capacity(),
                    sizeof(std::vector<Count>), "rank suffix-vector bytes"),
                "rank bytes");
            for (const auto& suffix : node.suffix) {
                stats_.suffix_cells = checked_size_add(
                    stats_.suffix_cells, suffix.capacity(),
                    "rank suffix cells");
                bytes = checked_size_add(bytes,
                    checked_size_multiply(suffix.capacity(), sizeof(Count),
                        "rank suffix bytes"),
                    "rank bytes");
            }
            for (const auto& group : node.groups) {
                stats_.prefix_cells = checked_size_add(
                    stats_.prefix_cells, group.cumulative.size(),
                    "rank prefix cells");
                bytes = checked_size_add(bytes,
                    checked_size_multiply(group.tuples.capacity(), sizeof(int),
                        "rank tuple-list bytes"),
                    "rank bytes");
                bytes = checked_size_add(bytes,
                    checked_size_multiply(group.cumulative.capacity(), sizeof(Count),
                        "rank prefix bytes"),
                    "rank bytes");
            }
        }
        stats_.approximate_bytes = bytes;
    }

    void access_group(int relation, Key key, Count rank, Output& output) const {
        const CountGroup* group = find_group(relation, key);
        if (!group || rank >= group->total)
            throw std::runtime_error("invalid join-rank group access");
        auto position_it = std::upper_bound(
            group->cumulative.begin(), group->cumulative.end(), rank);
        const std::size_t position = static_cast<std::size_t>(
            position_it - group->cumulative.begin());
        if (position >= group->tuples.size())
            throw std::logic_error("rank prefix search escaped compact group");
        const Count before = position == 0 ? 0 : group->cumulative[position - 1];
        Count local = rank - before;
        const int tuple_index = group->tuples[position];
        const Tuple& tuple = in_.rel[static_cast<std::size_t>(relation)].tuples[
            static_cast<std::size_t>(tuple_index)];
        output[static_cast<std::size_t>(relation)] = tuple.id;

        const auto& children = in_.children[static_cast<std::size_t>(relation)];
        const auto& suffix = node_[static_cast<std::size_t>(relation)].suffix[
            static_cast<std::size_t>(tuple_index)];
        for (std::size_t child_position = 0;
             child_position < children.size(); ++child_position) {
            const Count later = suffix[child_position + 1];
            const int child = children[child_position];
            const Key child_key =
                tuple.edge_key[static_cast<std::size_t>(child)];
            const CountGroup* child_group = find_group(child, child_key);
            if (!child_group || child_group->total == 0 || later == 0)
                throw std::logic_error("positive rank reached zero child group");
            const Count child_rank = local / later;
            local %= later;
            if (child_rank >= child_group->total)
                throw std::logic_error("rank Cartesian decomposition failed");
            access_group(child, child_key, child_rank, output);
        }
    }

    template<class Callback>
    void enumerate_group(std::size_t depth, Output& ids,
                         std::vector<int>& chosen, double probability,
                         Callback& callback,
                         EnumerationStats* enumeration) const {
        const int relation = in_.preorder[depth];
        Key key = ROOT_KEY;
        if (in_.parent[static_cast<std::size_t>(relation)] >= 0) {
            const int parent = in_.parent[static_cast<std::size_t>(relation)];
            key = in_.rel[static_cast<std::size_t>(parent)].tuples[
                static_cast<std::size_t>(chosen[static_cast<std::size_t>(parent)])]
                      .edge_key[static_cast<std::size_t>(relation)];
        }
        const CountGroup* group = find_group(relation, key);
        if (!group) return;
        for (int tuple_index : group->tuples) {
            if (enumeration)
                enumeration->tuple_extensions = checked_add(
                    enumeration->tuple_extensions, 1,
                    "grouped enumeration extensions");
            const Tuple& tuple = in_.rel[static_cast<std::size_t>(relation)].tuples[
                static_cast<std::size_t>(tuple_index)];
            chosen[static_cast<std::size_t>(relation)] = tuple_index;
            ids[static_cast<std::size_t>(relation)] = tuple.id;
            const double next_probability = probability * tuple.weight;
            if (depth + 1 == in_.preorder.size()) {
                if (enumeration)
                    enumeration->outputs = checked_add(
                        enumeration->outputs, 1,
                        "grouped enumeration outputs");
                callback(ids, next_probability);
            } else {
                enumerate_group(depth + 1, ids, chosen, next_probability,
                                callback, enumeration);
            }
        }
    }
};

}  

class UnsupportedUniformPopulation : public std::length_error {
public:
    explicit UnsupportedUniformPopulation(const std::string& message)
        : std::length_error(message) {}
};

struct UniformQueryStats {
    Count selected_ranks = 0;
};

struct UniformBuildStats {
    RankBuildStats rank;
    double output_probability = 0.0;
    std::size_t approximate_bytes = 0;
};

class UniformRankSampler {
public:
    static constexpr Count MAX_EXACT_DOUBLE_POPULATION = Count{1} << 53;

    explicit UniformRankSampler(const Instance& instance)
        : in_(instance), rank_(instance) {
        output_probability_ = validate_and_compute_probability();
        if (rank_.join_size() > MAX_EXACT_DOUBLE_POPULATION)
            throw UnsupportedUniformPopulation(
                "uniform rank control supports at most 2^53 join outputs");
        stats_.rank = rank_.build_stats();
        stats_.output_probability = output_probability_;
        stats_.approximate_bytes = checked_size_add(
            sizeof(*this) - sizeof(rank_), rank_.approximate_bytes(),
            "uniform rank sampler bytes");
    }

    UniformRankSampler(const UniformRankSampler&) = delete;
    UniformRankSampler& operator=(const UniformRankSampler&) = delete;

    Count join_size() const { return rank_.join_size(); }
    double output_probability() const { return output_probability_; }
    double expected_size() const {
        return static_cast<double>(join_size()) * output_probability_;
    }
    std::size_t approximate_bytes() const { return stats_.approximate_bytes; }
    const UniformBuildStats& build_stats() const { return stats_; }
    Output access_rank(Count rank) const { return rank_.access(rank); }

    std::vector<Output> sample_vector(std::mt19937_64& rng,
                                      UniformQueryStats& query_stats) const {
        std::vector<Output> result;
        const Count population = rank_.join_size();
        const double probability = output_probability_;
        if (population == 0 || !(probability > 0.0)) return result;
        if (probability >= 1.0) {
            result.reserve(static_cast<std::size_t>(population));
            for (Count rank = 0; rank < population; ++rank) {
                result.push_back(rank_.access(rank));
                query_stats.selected_ranks = checked_add(
                    query_stats.selected_ranks, 1,
                    "uniform selected ranks");
            }
            return result;
        }

        const double log_failure = std::log1p(-probability);
        if (!(log_failure < 0.0))
            throw std::runtime_error("invalid uniform geometric denominator");
        Count consumed = 0;
        while (consumed < population) {
            const Count remaining = population - consumed;
            const double real_failures =
                std::floor(std::log(uniform_open(rng)) / log_failure);
            if (!(real_failures >= 0.0))
                throw std::runtime_error("invalid uniform geometric gap");

            if (static_cast<long double>(real_failures) >=
                static_cast<long double>(remaining))
                break;
            const Count failures = static_cast<Count>(real_failures);
            const Count rank = consumed + failures;
            result.push_back(rank_.access(rank));
            query_stats.selected_ranks = checked_add(
                query_stats.selected_ranks, 1,
                "uniform selected ranks");
            consumed = rank + 1;
        }
        return result;
    }

    std::vector<Output> sample_vector(std::mt19937_64& rng) const {
        UniformQueryStats ignored;
        return sample_vector(rng, ignored);
    }

private:
    const Instance& in_;
    detail::CheckedRankIndex rank_;
    double output_probability_ = 0.0;
    UniformBuildStats stats_;

    double validate_and_compute_probability() const {
        bool empty_relation = false;
        std::vector<double> relation_weight(static_cast<std::size_t>(in_.m), 0.0);
        for (int relation = 0; relation < in_.m; ++relation) {
            const auto& tuples = in_.rel[static_cast<std::size_t>(relation)].tuples;
            if (tuples.empty()) {
                empty_relation = true;
                continue;
            }
            const double weight = tuples.front().weight;
            for (const Tuple& tuple : tuples) {
                if (tuple.weight != weight)
                    throw std::invalid_argument(
                        "UniformRankSampler requires a constant weight within each relation");
            }
            relation_weight[static_cast<std::size_t>(relation)] = weight;
        }
        if (empty_relation) return 0.0;
        double probability = 1.0;
        for (int relation : in_.preorder)
            probability *= relation_weight[static_cast<std::size_t>(relation)];
        if (!(probability >= 0.0 && probability <= 1.0) ||
            !std::isfinite(probability))
            throw std::runtime_error("invalid uniform output probability");
        return probability;
    }
};

class UnsupportedMaterialization : public std::length_error {
public:
    explicit UnsupportedMaterialization(const std::string& message)
        : std::length_error(message) {}
};

struct MaterializedOutput {
    Output ids;
    double probability = 0.0;
};

struct ScanQueryStats {
    Count rows_scanned = 0;
    Count outputs = 0;
};

struct ScanBuildStats {
    RankBuildStats rank;
    EnumerationStats enumeration;
    Count materialized_outputs = 0;
    double expected_size = 0.0;
    std::size_t approximate_bytes = 0;
};

class MaterializedScanSampler {
public:
    static constexpr Count MAX_MATERIALIZED_OUTPUTS = 1000000;

    explicit MaterializedScanSampler(const Instance& instance) {
        detail::CheckedRankIndex rank(instance);
        stats_.rank = rank.build_stats();
        if (rank.join_size() > MAX_MATERIALIZED_OUTPUTS)
            throw UnsupportedMaterialization(
                "materialized scan control supports at most 1,000,000 join outputs");
        rows_.reserve(static_cast<std::size_t>(rank.join_size()));
        rank.enumerate_weighted(
            [&](const Output& ids, double probability) {
                if (!(probability >= 0.0 && probability <= 1.0) ||
                    !std::isfinite(probability))
                    throw std::runtime_error(
                        "materialized output probability is invalid");
                rows_.push_back(MaterializedOutput{ids, probability});
                stats_.expected_size += probability;
            },
            &stats_.enumeration);
        if (rows_.size() != static_cast<std::size_t>(rank.join_size()) ||
            stats_.enumeration.outputs != rank.join_size())
            throw std::logic_error("grouped materialization count mismatch");
        const Count extension_limit = checked_multiply(
            static_cast<Count>(instance.m), rank.join_size(),
            "grouped enumeration mJ bound");
        if (stats_.enumeration.tuple_extensions > extension_limit)
            throw std::logic_error(
                "compacted grouped materialization exceeded mJ extensions");
        stats_.materialized_outputs = rank.join_size();
        finalize_bytes();
    }

    Count join_size() const { return stats_.materialized_outputs; }
    double expected_size() const { return stats_.expected_size; }
    std::size_t approximate_bytes() const { return stats_.approximate_bytes; }
    const ScanBuildStats& build_stats() const { return stats_; }
    const std::vector<MaterializedOutput>& rows() const { return rows_; }

    std::vector<Output> sample_vector(std::mt19937_64& rng,
                                      ScanQueryStats& query_stats) const {
        std::vector<Output> result;
        for (const auto& row : rows_) {
            query_stats.rows_scanned = checked_add(
                query_stats.rows_scanned, 1, "materialized rows scanned");
            if (coin(rng, row.probability)) {
                result.push_back(row.ids);
                query_stats.outputs = checked_add(
                    query_stats.outputs, 1, "materialized scan outputs");
            }
        }
        return result;
    }

    std::vector<Output> sample_vector(std::mt19937_64& rng) const {
        ScanQueryStats ignored;
        return sample_vector(rng, ignored);
    }

private:
    std::vector<MaterializedOutput> rows_;
    ScanBuildStats stats_;

    void finalize_bytes() {
        std::size_t bytes = sizeof(*this);
        bytes = checked_size_add(bytes,
            checked_size_multiply(rows_.capacity(), sizeof(MaterializedOutput),
                "materialized row bytes"),
            "materialized bytes");
        for (const auto& row : rows_)
            bytes = checked_size_add(bytes,
                checked_size_multiply(row.ids.capacity(), sizeof(int),
                    "materialized id bytes"),
                "materialized bytes");
        stats_.approximate_bytes = bytes;
    }
};

}
