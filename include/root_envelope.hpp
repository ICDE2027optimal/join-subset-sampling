// Root-block baseline with conditional binomial candidate counts.
#pragma once

#include "explicit_subset.hpp"
#include "uniform_envelope.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

namespace root_envelope {

using Count = uniform_envelope::Count;

struct QueryStats {
    Count selected_blocks = 0;
    Count positive_binomial_attempts = 0;
    Count binomial_draws = 0;
    Count sampled_candidates = 0;
    Count sparse_blocks = 0;
    Count dense_blocks = 0;
    Count wr_draws = 0;
    Count duplicate_wr_draws = 0;
    Count distinct_candidates = 0;
    Count enumerated_results = 0;
    Count thinning_coins = 0;
    Count outputs = 0;
};

struct BuildStats {
    std::size_t root_blocks = 0;
    std::size_t active_root_blocks = 0;
    Count join_size = 0;
    double mu = 0.0;
    double lambda_root = 0.0;
    double expected_selected_blocks = 0.0;

    std::size_t approximate_bytes_excluding_meta = 0;
    std::size_t explicit_meta_bytes = 0;
    std::size_t approximate_bytes = 0;
    bool double_alias_is_mathematically_exact = false;
};

class RootStratifiedEnvelopeIndex {
public:
    explicit RootStratifiedEnvelopeIndex(const Instance& in)
        : base_(in) {
        build_meta();
    }
    RootStratifiedEnvelopeIndex(Instance&&) = delete;

    Count join_size() const { return base_.join_size(); }
    double mu() const { return base_.mu(); }
    double lambda_root() const { return build_stats_.lambda_root; }
    const BuildStats& build_stats() const { return build_stats_; }
    std::size_t approximate_bytes() const {
        return build_stats_.approximate_bytes;
    }
    const uniform_envelope::UniformEnvelopeIndex& completion_index() const {
        return base_;
    }
    const std::vector<double>& block_nonempty_probabilities() const {
        return block_nonempty_probability_;
    }

    std::vector<Output> sample_vector(std::mt19937_64& rng,
                                      QueryStats& stats) const {
        std::vector<Output> output;
        const std::vector<std::size_t> selected = meta_.query(rng);
        stats.selected_blocks = checked_add(
            stats.selected_blocks, static_cast<Count>(selected.size()),
            "selected root blocks");
        for (std::size_t block_index : selected) {
            const auto& block = base_.root_blocks().at(block_index);
            if (block.join_size == 0 || !(block.envelope > 0.0))
                throw std::logic_error("meta sampler selected an empty root block");
            const Count k = positive_binomial(block, rng, stats);
            append_block_sample(block_index, k, rng, stats, output, true);
        }
        return output;
    }

    std::vector<std::size_t>
    sample_meta_blocks_for_validation(std::mt19937_64& rng) const {
        return meta_.query(rng);
    }

    Count positive_binomial_for_validation(std::size_t block_index,
                                           std::mt19937_64& rng,
                                           QueryStats& stats) const {
        return positive_binomial(base_.root_blocks().at(block_index), rng, stats);
    }

    std::vector<Output>
    uniform_k_subset_for_validation(std::size_t block_index, Count k,
                                    std::mt19937_64& rng,
                                    QueryStats& stats) const {
        std::vector<Output> candidates;
        append_block_sample(block_index, k, rng, stats, candidates, false);
        return candidates;
    }

private:
    uniform_envelope::UniformEnvelopeIndex base_;
    std::vector<double> block_nonempty_probability_;
    poisson_join::ExplicitSubsetIndex meta_;
    BuildStats build_stats_;

    static Count checked_add(Count a, Count b, const char* where) {
        return uniform_envelope::checked_add(a, b, where);
    }

    static double at_least_one(Count count, double probability) {
        if (count == 0 || probability <= 0.0) return 0.0;
        if (probability >= 1.0) return 1.0;
        const long double exponent =
            static_cast<long double>(count) *
            std::log1p(-static_cast<long double>(probability));
        const long double value = -std::expm1(exponent);
        const double result = static_cast<double>(value);
        if (!(result > 0.0 && result <= 1.0))
            throw std::runtime_error("invalid root-block nonempty probability");
        return result;
    }

    void build_meta() {
        const auto& blocks = base_.root_blocks();
        block_nonempty_probability_.reserve(blocks.size());
        long double lambda = 0.0L;
        std::size_t active = 0;
        for (const auto& block : blocks) {
            const double q = at_least_one(block.join_size, block.envelope);
            block_nonempty_probability_.push_back(q);
            if (q > 0.0) ++active;
            lambda += static_cast<long double>(block.join_size) *
                      static_cast<long double>(block.envelope);
        }
        meta_.build(block_nonempty_probability_);
        build_stats_.root_blocks = blocks.size();
        build_stats_.active_root_blocks = active;
        build_stats_.join_size = base_.join_size();
        build_stats_.mu = base_.mu();
        build_stats_.lambda_root = static_cast<double>(lambda);
        build_stats_.expected_selected_blocks = meta_.expected_size();
        build_stats_.approximate_bytes_excluding_meta =
            base_.approximate_bytes() +
            block_nonempty_probability_.capacity() * sizeof(double);
        build_stats_.explicit_meta_bytes = meta_.approximate_bytes();
        build_stats_.approximate_bytes =
            build_stats_.approximate_bytes_excluding_meta +
            build_stats_.explicit_meta_bytes;
        build_stats_.double_alias_is_mathematically_exact = false;
    }

    static Count ordinary_binomial(Count count, double probability,
                                   std::mt19937_64& rng,
                                   QueryStats& stats) {
        if (count == 0 || probability <= 0.0) return 0;
        if (probability >= 1.0) return count;
        ++stats.binomial_draws;
        if (probability <= 0.5) {
            std::binomial_distribution<unsigned long long> distribution(
                static_cast<unsigned long long>(count), probability);
            return static_cast<Count>(distribution(rng));
        }
        std::binomial_distribution<unsigned long long> failures(
            static_cast<unsigned long long>(count), 1.0 - probability);
        return count - static_cast<Count>(failures(rng));
    }

    static Count positive_binomial(
        const uniform_envelope::RootBlockInfo& block,
        std::mt19937_64& rng, QueryStats& stats) {
        const Count count = block.join_size;
        const double upper = block.envelope;
        if (count == 0 || !(upper > 0.0) || upper > 1.0)
            throw std::invalid_argument("positive binomial needs J>0 and U in (0,1]");
        if (upper >= 1.0) {
            ++stats.positive_binomial_attempts;
            return count;
        }
        const long double lambda =
            static_cast<long double>(count) * static_cast<long double>(upper);
        if (lambda <= 1.0L) {
            while (true) {
                ++stats.positive_binomial_attempts;
                const Count k = 1 + ordinary_binomial(count - 1, upper, rng, stats);
                if (coin(rng, 1.0 / static_cast<double>(k))) return k;
            }
        }
        while (true) {
            ++stats.positive_binomial_attempts;
            const Count k = ordinary_binomial(count, upper, rng, stats);
            if (k != 0) return k;
        }
    }

    static bool fixed_size_coin(std::mt19937_64& rng, Count selected,
                                Count remaining) {
        if (selected == 0) return false;
        if (remaining == 0 || selected > remaining)
            throw std::logic_error("invalid root fixed-size selection state");
        if (selected == remaining) return true;
        std::uniform_int_distribution<Count> distribution(0, remaining - 1);
        return distribution(rng) < selected;
    }

    static void maybe_keep(const Output& ids, double probability, double upper,
                           std::mt19937_64& rng, QueryStats& stats,
                           std::vector<Output>& output, bool thin) {
        if (!thin) {
            output.push_back(ids);
            return;
        }
        ++stats.thinning_coins;
        const double keep = probability / upper;
        if (!(keep >= 0.0 && keep <= 1.0))
            throw std::runtime_error("root-envelope thinning probability outside [0,1]");
        if (coin(rng, keep)) {
            output.push_back(ids);
            ++stats.outputs;
        }
    }

    void append_block_sample(std::size_t block_index, Count k,
                             std::mt19937_64& rng, QueryStats& stats,
                             std::vector<Output>& output, bool thin) const {
        const auto& block = base_.root_blocks().at(block_index);
        if (k > block.join_size)
            throw std::out_of_range("root fixed K exceeds block size");
        if (k == 0) return;
        if (!(block.envelope > 0.0))
            throw std::invalid_argument("positive root fixed K with zero envelope");
        stats.sampled_candidates = checked_add(
            stats.sampled_candidates, k, "root sampled candidates");
        if (k <= block.join_size / 2) {
            ++stats.sparse_blocks;
            if (k > static_cast<Count>(std::numeric_limits<std::size_t>::max()))
                throw std::length_error("root candidate set exceeds size_t");
            const std::size_t target = static_cast<std::size_t>(k);
            if (target > (std::numeric_limits<std::size_t>::max() - 1) / 2)
                throw std::length_error("root candidate reserve overflow");
            std::unordered_set<Output, VecHash> candidates;
            candidates.reserve(target * 2 + 1);
            Output ids;
            std::vector<int> chosen;
            while (candidates.size() < target) {
                const double probability =
                    base_.uniform_draw_root_into(block_index, rng, ids, chosen);
                ++stats.wr_draws;
                auto [it, inserted] = candidates.insert(ids);
                if (!inserted) {
                    ++stats.duplicate_wr_draws;
                    continue;
                }
                ++stats.distinct_candidates;
                maybe_keep(*it, probability, block.envelope, rng, stats,
                           output, thin);
            }
            return;
        }

        ++stats.dense_blocks;
        Count remaining = block.join_size;
        Count selected = k;
        base_.enumerate_root_weighted(
            block_index, [&](const Output& ids, double probability) {
                ++stats.enumerated_results;
                if (fixed_size_coin(rng, selected, remaining)) {
                    --selected;
                    ++stats.distinct_candidates;
                    maybe_keep(ids, probability, block.envelope, rng, stats,
                               output, thin);
                }
                --remaining;
            });
        if (remaining != 0 || selected != 0)
            throw std::runtime_error("root fixed-size enumeration did not consume K");
    }
};

}
