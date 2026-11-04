// Full witness storage combined with an explicit subset index.
#pragma once

#include "explicit_subset.hpp"
#include "rank_scan_controls.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace materialized_explicit {

using Count = rank_scan_controls::Count;

class UnsupportedMaterializedExplicit : public std::length_error {
public:
    explicit UnsupportedMaterializedExplicit(const std::string& message)
        : std::length_error(message) {}
};

struct BuildStats {
    rank_scan_controls::RankBuildStats rank;
    rank_scan_controls::EnumerationStats enumeration;
    Count materialized_outputs = 0;
    std::size_t relations = 0;

    std::size_t flat_id_cells = 0;
    std::size_t flat_id_capacity = 0;
    std::size_t flat_id_bytes = 0;

    long double expected_size = 0.0L;
    double max_probability = 0.0;
    long double sum_p2 = 0.0L;

    std::size_t subset_bucket_count = 0;
    std::size_t subset_tail_size = 0;
    std::size_t subset_index_bytes = 0;
    std::size_t subset_dynamic_bytes = 0;

    std::size_t temporary_probability_cells = 0;
    std::size_t temporary_probability_capacity = 0;
    std::size_t temporary_probability_bytes = 0;

    std::size_t owned_persistent_bytes = 0;
};

struct QueryStats {
    Count selected_indices = 0;
    Count outputs = 0;
    std::size_t selected_index_temporary_bytes = 0;
};

class MaterializedExplicitSampler {
public:
    
    static constexpr Count MAX_MATERIALIZED_OUTPUTS = 25000000;

    explicit MaterializedExplicitSampler(const Instance& instance) {
        std::vector<double> probabilities;
        {
            
            rank_scan_controls::detail::CheckedRankIndex rank(instance);
            relation_count_ = static_cast<std::size_t>(instance.m);
            stats_.rank = rank.build_stats();
            join_size_ = rank.join_size();
            if (join_size_ > MAX_MATERIALIZED_OUTPUTS)
                throw UnsupportedMaterializedExplicit(
                    "materialized explicit control exceeds the 25,000,000-output study budget");
            if constexpr (sizeof(std::size_t) < sizeof(Count)) {
                if (join_size_ > static_cast<Count>(
                        std::numeric_limits<std::size_t>::max()))
                    throw UnsupportedMaterializedExplicit(
                        "materialized explicit output count exceeds size_t");
            }

            const std::size_t outputs = static_cast<std::size_t>(join_size_);
            const std::size_t id_cells = checked_size_multiply(
                outputs, relation_count_, "flat output ID cells");
            if (id_cells > static_cast<std::size_t>(
                    std::numeric_limits<std::ptrdiff_t>::max()))
                throw UnsupportedMaterializedExplicit(
                    "flat output ID cells exceed iterator difference range");
            flat_ids_.reserve(id_cells);
            probabilities.reserve(outputs);

            rank.enumerate_weighted(
                [&](const Output& ids, double probability) {
                    if (ids.size() != relation_count_)
                        throw std::logic_error(
                            "materialized explicit enumeration returned wrong output arity");
                    if (!std::isfinite(probability) || probability < 0.0 ||
                        probability > 1.0)
                        throw std::runtime_error(
                            "materialized explicit output probability is invalid");
                    flat_ids_.insert(flat_ids_.end(), ids.begin(), ids.end());
                    probabilities.push_back(probability);
                    stats_.expected_size += static_cast<long double>(probability);
                    stats_.sum_p2 += static_cast<long double>(probability) *
                                     static_cast<long double>(probability);
                    stats_.max_probability =
                        std::max(stats_.max_probability, probability);
                },
                &stats_.enumeration);
        }  

        const std::size_t outputs = static_cast<std::size_t>(join_size_);
        const std::size_t expected_cells = checked_size_multiply(
            outputs, relation_count_, "expected flat output ID cells");
        if (probabilities.size() != outputs || flat_ids_.size() != expected_cells ||
            stats_.enumeration.outputs != join_size_)
            throw std::logic_error("materialized explicit enumeration count mismatch");
        const Count extension_limit = rank_scan_controls::checked_multiply(
            static_cast<Count>(relation_count_), join_size_,
            "materialized explicit grouped enumeration mJ bound");
        if (stats_.enumeration.tuple_extensions > extension_limit)
            throw std::logic_error(
                "compacted grouped materialization exceeded mJ extensions");

        stats_.temporary_probability_cells = probabilities.size();
        stats_.temporary_probability_capacity = probabilities.capacity();
        stats_.temporary_probability_bytes = checked_size_multiply(
            probabilities.capacity(), sizeof(double),
            "temporary output probability bytes");
        subset_.build(probabilities);
        if (subset_.size() != outputs)
            throw std::logic_error("explicit subset population mismatch");

        std::vector<double>().swap(probabilities);
        finalize_stats();
    }

    MaterializedExplicitSampler(const MaterializedExplicitSampler&) = delete;
    MaterializedExplicitSampler& operator=(const MaterializedExplicitSampler&) = delete;

    Count join_size() const noexcept { return join_size_; }
    std::size_t relation_count() const noexcept { return relation_count_; }
    long double mu() const noexcept { return stats_.expected_size; }
    double max_probability() const noexcept { return stats_.max_probability; }
    long double sum_p2() const noexcept { return stats_.sum_p2; }
    std::size_t bucket_count() const noexcept { return subset_.bucket_count(); }
    std::size_t tail_size() const noexcept { return subset_.tail_size(); }
    std::size_t approximate_bytes() const noexcept {
        return stats_.owned_persistent_bytes;
    }
    const BuildStats& build_stats() const noexcept { return stats_; }

    Output access_output(Count index) const {
        if (index >= join_size_)
            throw std::out_of_range("materialized explicit output index");
        return copy_output(static_cast<std::size_t>(index));
    }

    std::vector<Output> sample_vector(std::mt19937_64& rng,
                                      QueryStats& query_stats) const {
        std::vector<std::size_t> selected = subset_.query(rng);
        query_stats.selected_indices = rank_scan_controls::checked_add(
            query_stats.selected_indices, static_cast<Count>(selected.size()),
            "materialized explicit selected indices");
        query_stats.selected_index_temporary_bytes = std::max(
            query_stats.selected_index_temporary_bytes,
            checked_size_multiply(selected.capacity(), sizeof(std::size_t),
                                  "selected-index temporary bytes"));

        std::vector<Output> result;
        result.reserve(selected.size());
        for (std::size_t index : selected) {
            if (index >= static_cast<std::size_t>(join_size_))
                throw std::logic_error("explicit subset returned output index out of range");
            result.push_back(copy_output(index));
            query_stats.outputs = rank_scan_controls::checked_add(
                query_stats.outputs, 1, "materialized explicit query outputs");
        }

        return result;
    }

    std::vector<Output> sample_vector(std::mt19937_64& rng) const {
        QueryStats ignored;
        return sample_vector(rng, ignored);
    }

private:
    Count join_size_ = 0;
    std::size_t relation_count_ = 0;
    std::vector<int> flat_ids_;
    poisson_join::ExplicitSubsetIndex subset_;
    BuildStats stats_;

    static std::size_t checked_size_add(std::size_t a, std::size_t b,
                                        const char* where) {
        return rank_scan_controls::checked_size_add(a, b, where);
    }

    static std::size_t checked_size_multiply(std::size_t a, std::size_t b,
                                             const char* where) {
        return rank_scan_controls::checked_size_multiply(a, b, where);
    }

    Output copy_output(std::size_t index) const {
        const std::size_t offset = checked_size_multiply(
            index, relation_count_, "materialized explicit output offset");
        if (offset > flat_ids_.size() ||
            relation_count_ > flat_ids_.size() - offset)
            throw std::logic_error("materialized explicit flat output bounds");
        return Output(flat_ids_.begin() + static_cast<std::ptrdiff_t>(offset),
                      flat_ids_.begin() + static_cast<std::ptrdiff_t>(
                          offset + relation_count_));
    }

    void finalize_stats() {
        stats_.materialized_outputs = join_size_;
        stats_.relations = relation_count_;
        stats_.flat_id_cells = flat_ids_.size();
        stats_.flat_id_capacity = flat_ids_.capacity();
        stats_.flat_id_bytes = checked_size_multiply(
            flat_ids_.capacity(), sizeof(int), "persistent flat output ID bytes");
        stats_.subset_bucket_count = subset_.bucket_count();
        stats_.subset_tail_size = subset_.tail_size();
        stats_.subset_index_bytes = subset_.approximate_bytes();
        if (stats_.subset_index_bytes < sizeof(poisson_join::ExplicitSubsetIndex))
            throw std::logic_error("explicit subset byte accounting below object size");
        stats_.subset_dynamic_bytes =
            stats_.subset_index_bytes - sizeof(poisson_join::ExplicitSubsetIndex);

        std::size_t bytes = sizeof(*this);
        bytes = checked_size_add(bytes, stats_.flat_id_bytes,
                                 "materialized explicit persistent bytes");
        bytes = checked_size_add(bytes, stats_.subset_dynamic_bytes,
                                 "materialized explicit persistent bytes");
        stats_.owned_persistent_bytes = bytes;
    }
};

}
