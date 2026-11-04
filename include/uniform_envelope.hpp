// Global-envelope baseline with binary64 alias draws from integer counts.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace uniform_envelope {

using Count = std::uint64_t;
static constexpr Count MAX_EXACT_DOUBLE_INTEGER = (Count{1} << 53);

inline Count checked_add(Count a, Count b, const char* where) {
    if (std::numeric_limits<Count>::max() - a < b)
        throw std::overflow_error(std::string("uint64 count addition overflow at ") + where);
    return a + b;
}

inline Count checked_mul(Count a, Count b, const char* where) {
    if (a != 0 && b > std::numeric_limits<Count>::max() / a)
        throw std::overflow_error(std::string("uint64 count multiplication overflow at ") + where);
    return a * b;
}

struct EnvelopeGroup {
    Key key = 0;
    std::vector<int> tuples;
    Count completions = 0;
    double probability_sum = 0.0;
    long double max_product = 0.0L;
    AliasTable uniform_alias;
};

struct EnvelopeNode {
    std::unordered_map<Key, std::size_t> lookup;
    std::vector<EnvelopeGroup> groups;
    std::vector<Count> tuple_completions;
    std::vector<double> tuple_probability_sum;
    std::vector<long double> tuple_max_product;
};

struct BuildStats {
    std::size_t input_tuples = 0;
    std::size_t groups = 0;
    std::size_t alias_entries = 0;
    Count join_size = 0;
    double mu = 0.0;
    double max_product_dp = 0.0;
    double envelope = 0.0;
    bool counts_within_exact_double_integer_range = true;
    bool double_alias_is_mathematically_exact = false;
    std::size_t approximate_bytes = 0;
};

struct QueryStats {
    Count binomial_draws = 0;
    Count sampled_k = 0;
    Count sparse_queries = 0;
    Count dense_queries = 0;
    Count wr_draws = 0;
    Count duplicate_wr_draws = 0;
    Count distinct_candidates = 0;
    Count enumerated_results = 0;
    Count thinning_coins = 0;
    Count outputs = 0;
};

struct RootBlockInfo {
    std::size_t block_index = 0;
    int root_tuple_index = -1;
    int root_tuple_id = -1;
    Count join_size = 0;
    double mu = 0.0;
    double max_probability_dp = 0.0;
    double envelope = 0.0;
};

class UniformEnvelopeIndex {
public:
    explicit UniformEnvelopeIndex(const Instance& in)
        : in_(in), node_(static_cast<std::size_t>(in.m)) {
        if (in_.m <= 0 || in_.preorder.empty())
            throw std::invalid_argument("uniform envelope needs a nonempty join tree");
        build();
    }

    UniformEnvelopeIndex(const UniformEnvelopeIndex&) = delete;
    UniformEnvelopeIndex& operator=(const UniformEnvelopeIndex&) = delete;
    UniformEnvelopeIndex(UniformEnvelopeIndex&&) = delete;
    UniformEnvelopeIndex& operator=(UniformEnvelopeIndex&&) = delete;

    Count join_size() const { return join_size_; }
    double mu() const { return mu_; }
    double max_probability() const { return max_probability_dp_; }
    double envelope_probability() const { return envelope_; }
    double kappa() const {
        if (!(mu_ > 0.0)) return mu_ == 0.0 && envelope_ == 0.0
            ? 1.0 : std::numeric_limits<double>::infinity();
        return static_cast<double>(join_size_) * envelope_ / mu_;
    }
    const BuildStats& build_stats() const { return build_stats_; }
    std::size_t approximate_bytes() const { return build_stats_.approximate_bytes; }
    const std::vector<RootBlockInfo>& root_blocks() const { return root_blocks_; }

    double uniform_draw_into(std::mt19937_64& rng, Output& ids,
                             std::vector<int>& chosen) const {
        if (join_size_ == 0) throw std::runtime_error("uniform draw on empty join");
        if (ids.size() != static_cast<std::size_t>(in_.m))
            ids.resize(static_cast<std::size_t>(in_.m));
        if (chosen.size() != static_cast<std::size_t>(in_.m))
            chosen.resize(static_cast<std::size_t>(in_.m));
        double probability = 1.0;
        for (int v : in_.preorder) {
            const EnvelopeGroup* group = nullptr;
            if (in_.parent[static_cast<std::size_t>(v)] < 0) {
                group = find_group(v, ROOT_KEY);
            } else {
                const int parent = in_.parent[static_cast<std::size_t>(v)];
                const Tuple& parent_tuple =
                    in_.rel[static_cast<std::size_t>(parent)].tuples[
                        static_cast<std::size_t>(chosen[static_cast<std::size_t>(parent)])];
                group = find_group(v, parent_tuple.edge_key[static_cast<std::size_t>(v)]);
            }
            if (!group || group->completions == 0 || group->uniform_alias.empty())
                throw std::runtime_error("uniform draw reached empty completion group");
            const int ti = group->uniform_alias.sample(rng);
            chosen[static_cast<std::size_t>(v)] = ti;
            const Tuple& tuple =
                in_.rel[static_cast<std::size_t>(v)].tuples[static_cast<std::size_t>(ti)];
            ids[static_cast<std::size_t>(v)] = tuple.id;
            probability *= tuple.weight;
        }
        verify_envelope(probability);
        return probability;
    }

    std::pair<Output, double> uniform_draw(std::mt19937_64& rng) const {
        Output ids;
        std::vector<int> chosen;
        const double p = uniform_draw_into(rng, ids, chosen);
        return {std::move(ids), p};
    }

    double uniform_draw_root_into(std::size_t block_index,
                                  std::mt19937_64& rng, Output& ids,
                                  std::vector<int>& chosen) const {
        const RootBlockInfo& block = checked_root_block(block_index);
        if (block.join_size == 0)
            throw std::runtime_error("uniform draw on empty root block");
        if (ids.size() != static_cast<std::size_t>(in_.m))
            ids.resize(static_cast<std::size_t>(in_.m));
        if (chosen.size() != static_cast<std::size_t>(in_.m))
            chosen.resize(static_cast<std::size_t>(in_.m));
        const int root = in_.preorder.front();
        const int ti = block.root_tuple_index;
        const Tuple& root_tuple =
            in_.rel[static_cast<std::size_t>(root)].tuples[static_cast<std::size_t>(ti)];
        chosen[static_cast<std::size_t>(root)] = ti;
        ids[static_cast<std::size_t>(root)] = root_tuple.id;
        double probability = root_tuple.weight;
        probability = draw_from_preorder_depth(1, rng, ids, chosen, probability);
        verify_envelope(probability, block.envelope);
        return probability;
    }

    std::pair<Output, double>
    uniform_draw_root(std::size_t block_index, std::mt19937_64& rng) const {
        Output ids;
        std::vector<int> chosen;
        const double p =
            uniform_draw_root_into(block_index, rng, ids, chosen);
        return {std::move(ids), p};
    }

    template<class Callback>
    void enumerate_weighted(Callback&& callback) const {
        if (join_size_ == 0) return;
        Output ids(static_cast<std::size_t>(in_.m), -1);
        std::vector<int> chosen(static_cast<std::size_t>(in_.m), -1);
        enumerate_depth(0, ids, chosen, 1.0, callback);
    }

    template<class Callback>
    void enumerate_root_weighted(std::size_t block_index,
                                 Callback&& callback) const {
        const RootBlockInfo& block = checked_root_block(block_index);
        if (block.join_size == 0) return;
        Output ids(static_cast<std::size_t>(in_.m), -1);
        std::vector<int> chosen(static_cast<std::size_t>(in_.m), -1);
        const int root = in_.preorder.front();
        const int ti = block.root_tuple_index;
        const Tuple& tuple =
            in_.rel[static_cast<std::size_t>(root)].tuples[static_cast<std::size_t>(ti)];
        chosen[static_cast<std::size_t>(root)] = ti;
        ids[static_cast<std::size_t>(root)] = tuple.id;
        if (in_.preorder.size() == 1) {
            verify_envelope(tuple.weight, block.envelope);
            callback(ids, tuple.weight);
        } else {
            auto checked_callback = [&](const Output& output, double probability) {
                verify_envelope(probability, block.envelope);
                callback(output, probability);
            };
            enumerate_depth(1, ids, chosen, tuple.weight, checked_callback);
        }
    }

    std::vector<Output> sample_vector(std::mt19937_64& rng,
                                      QueryStats& stats) const {
        if (join_size_ == 0 || envelope_ == 0.0) return {};
        Count k = 0;
        if (envelope_ >= 1.0) {
            k = join_size_;
        } else {
            ++stats.binomial_draws;
            if (envelope_ <= 0.5) {
                std::binomial_distribution<unsigned long long> distribution(
                    static_cast<unsigned long long>(join_size_), envelope_);
                k = static_cast<Count>(distribution(rng));
            } else {
                std::binomial_distribution<unsigned long long> distribution(
                    static_cast<unsigned long long>(join_size_), 1.0 - envelope_);
                const Count failures = static_cast<Count>(distribution(rng));
                k = join_size_ - failures;
            }
        }
        return sample_vector_given_k(rng, k, stats);
    }

    std::vector<Output> sample_vector_given_k(std::mt19937_64& rng, Count k,
                                              QueryStats& stats) const {
        if (k > join_size_) throw std::out_of_range("fixed K exceeds join size");
        stats.sampled_k = checked_add(stats.sampled_k, k, "query sampled K");
        if (k == 0) {
            ++stats.sparse_queries;
            return {};
        }
        if (!(envelope_ > 0.0))
            throw std::invalid_argument("positive fixed K with zero envelope");
        if (k <= join_size_ / 2) return sample_sparse(rng, k, stats);
        return sample_dense(rng, k, stats);
    }

private:
    const Instance& in_;
    std::vector<EnvelopeNode> node_;
    Count join_size_ = 0;
    double mu_ = 0.0;
    double max_probability_dp_ = 0.0;
    double envelope_ = 0.0;
    BuildStats build_stats_;
    std::vector<RootBlockInfo> root_blocks_;

    EnvelopeGroup* find_group_mut(int v, Key key) {
        auto& nd = node_[static_cast<std::size_t>(v)];
        auto it = nd.lookup.find(key);
        return it == nd.lookup.end() ? nullptr : &nd.groups[it->second];
    }

    const EnvelopeGroup* find_group(int v, Key key) const {
        const auto& nd = node_[static_cast<std::size_t>(v)];
        auto it = nd.lookup.find(key);
        return it == nd.lookup.end() ? nullptr : &nd.groups[it->second];
    }

    void build() {
        for (int v = 0; v < in_.m; ++v) {
            auto& nd = node_[static_cast<std::size_t>(v)];
            const auto& tuples = in_.rel[static_cast<std::size_t>(v)].tuples;
            nd.tuple_completions.assign(tuples.size(), 0);
            nd.tuple_probability_sum.assign(tuples.size(), 0.0);
            nd.tuple_max_product.assign(tuples.size(), 0.0L);
            build_stats_.input_tuples += tuples.size();
            for (std::size_t ti = 0; ti < tuples.size(); ++ti) {
                const Tuple& tuple = tuples[ti];
                if (!(tuple.weight >= 0.0 && tuple.weight <= 1.0) ||
                    std::isnan(tuple.weight))
                    throw std::invalid_argument("tuple weight must lie in [0,1]");
                const Key key = in_.parent[static_cast<std::size_t>(v)] < 0
                    ? ROOT_KEY
                    : tuple.edge_key[static_cast<std::size_t>(
                        in_.parent[static_cast<std::size_t>(v)])];
                auto [it, inserted] = nd.lookup.emplace(key, nd.groups.size());
                if (inserted) {
                    nd.groups.push_back(EnvelopeGroup{});
                    nd.groups.back().key = key;
                }
                nd.groups[it->second].tuples.push_back(static_cast<int>(ti));
            }
        }

        for (auto order = in_.preorder.rbegin(); order != in_.preorder.rend(); ++order) {
            const int v = *order;
            auto& nd = node_[static_cast<std::size_t>(v)];
            const auto& tuples = in_.rel[static_cast<std::size_t>(v)].tuples;
            for (std::size_t ti = 0; ti < tuples.size(); ++ti) {
                const Tuple& tuple = tuples[ti];
                Count completions = 1;
                double sum = tuple.weight;
                long double maximum = static_cast<long double>(tuple.weight);
                for (int child : in_.children[static_cast<std::size_t>(v)]) {
                    const EnvelopeGroup* group =
                        find_group(child, tuple.edge_key[static_cast<std::size_t>(child)]);
                    if (!group || group->completions == 0) {
                        completions = 0;
                        sum = 0.0;
                        maximum = 0.0L;
                        break;
                    }
                    completions = checked_mul(completions, group->completions,
                                              "tuple completion product");
                    sum *= group->probability_sum;
                    maximum *= group->max_product;
                }
                nd.tuple_completions[ti] = completions;
                nd.tuple_probability_sum[ti] = sum;
                nd.tuple_max_product[ti] = maximum;
            }

            for (auto& group : nd.groups) {
                std::vector<double> alias_weight;
                alias_weight.reserve(group.tuples.size());
                for (int ti : group.tuples) {
                    const std::size_t index = static_cast<std::size_t>(ti);
                    group.completions = checked_add(
                        group.completions, nd.tuple_completions[index],
                        "group completion sum");
                    group.probability_sum += nd.tuple_probability_sum[index];
                    group.max_product =
                        std::max(group.max_product, nd.tuple_max_product[index]);
                    alias_weight.push_back(
                        static_cast<double>(nd.tuple_completions[index]));
                }
                if (group.completions > MAX_EXACT_DOUBLE_INTEGER)
                    throw std::overflow_error(
                        "double count alias requires every group count <= 2^53");
                group.uniform_alias.build(group.tuples, alias_weight);
            }
        }

        const EnvelopeGroup* root = find_group(in_.preorder.front(), ROOT_KEY);
        if (root) {
            join_size_ = root->completions;
            mu_ = root->probability_sum;
            max_probability_dp_ = static_cast<double>(root->max_product);
            if (root->max_product > 0.0L && max_probability_dp_ == 0.0)
                throw std::underflow_error("max-product envelope underflowed double");
        }
        if (max_probability_dp_ <= 0.0) {
            envelope_ = 0.0;
        } else if (max_probability_dp_ >= 1.0) {
            envelope_ = 1.0;
        } else {
            const double eps = std::numeric_limits<double>::epsilon();
            const double inflation =
                1.0 + 64.0 * static_cast<double>(in_.m) * eps;
            const double inflated = max_probability_dp_ * inflation;
            envelope_ = std::min(
                1.0, std::nextafter(inflated,
                                    std::numeric_limits<double>::infinity()));
        }
        if (!(envelope_ >= 0.0 && envelope_ <= 1.0))
            throw std::runtime_error("invalid implemented envelope");
        build_root_blocks();
        finalize_build_stats();
    }

    double guarded_envelope(double maximum) const {
        if (maximum <= 0.0) return 0.0;
        if (maximum >= 1.0) return 1.0;
        const double eps = std::numeric_limits<double>::epsilon();
        const double inflated =
            maximum * (1.0 + 64.0 * static_cast<double>(in_.m) * eps);
        return std::min(
            1.0, std::nextafter(inflated,
                                std::numeric_limits<double>::infinity()));
    }

    void build_root_blocks() {
        const int root = in_.preorder.front();
        const auto& tuples = in_.rel[static_cast<std::size_t>(root)].tuples;
        const auto& nd = node_[static_cast<std::size_t>(root)];
        root_blocks_.reserve(tuples.size());
        for (std::size_t ti = 0; ti < tuples.size(); ++ti) {
            const long double maximum_ld = nd.tuple_max_product[ti];
            const double maximum = static_cast<double>(maximum_ld);
            if (maximum_ld > 0.0L && maximum == 0.0)
                throw std::underflow_error("root-block max product underflowed double");
            root_blocks_.push_back(RootBlockInfo{
                ti, static_cast<int>(ti), tuples[ti].id,
                nd.tuple_completions[ti], nd.tuple_probability_sum[ti],
                maximum, guarded_envelope(maximum)});
        }
    }

    const RootBlockInfo& checked_root_block(std::size_t block_index) const {
        if (block_index >= root_blocks_.size())
            throw std::out_of_range("root block index");
        return root_blocks_[block_index];
    }

    void verify_envelope(double probability) const {
        verify_envelope(probability, envelope_);
    }

    static void verify_envelope(double probability, double envelope) {
        if (!(probability >= 0.0 && probability <= envelope))
            throw std::runtime_error(
                "output probability exceeds guarded uniform envelope");
    }

    double draw_from_preorder_depth(std::size_t depth, std::mt19937_64& rng,
                                    Output& ids, std::vector<int>& chosen,
                                    double probability) const {
        for (std::size_t order = depth; order < in_.preorder.size(); ++order) {
            const int v = in_.preorder[order];
            const int parent = in_.parent[static_cast<std::size_t>(v)];
            if (parent < 0)
                throw std::logic_error("second root in relation tree");
            const Tuple& parent_tuple =
                in_.rel[static_cast<std::size_t>(parent)].tuples[
                    static_cast<std::size_t>(chosen[static_cast<std::size_t>(parent)])];
            const EnvelopeGroup* group =
                find_group(v, parent_tuple.edge_key[static_cast<std::size_t>(v)]);
            if (!group || group->completions == 0 || group->uniform_alias.empty())
                throw std::runtime_error("root-block draw reached empty group");
            const int ti = group->uniform_alias.sample(rng);
            chosen[static_cast<std::size_t>(v)] = ti;
            const Tuple& tuple =
                in_.rel[static_cast<std::size_t>(v)].tuples[static_cast<std::size_t>(ti)];
            ids[static_cast<std::size_t>(v)] = tuple.id;
            probability *= tuple.weight;
        }
        return probability;
    }

    template<class Callback>
    void enumerate_depth(std::size_t depth, Output& ids,
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
        const EnvelopeGroup* group = find_group(v, key);
        if (!group) return;
        for (int ti : group->tuples) {
            if (node_[static_cast<std::size_t>(v)].tuple_completions[
                    static_cast<std::size_t>(ti)] == 0)
                continue;
            const Tuple& tuple =
                in_.rel[static_cast<std::size_t>(v)].tuples[static_cast<std::size_t>(ti)];
            chosen[static_cast<std::size_t>(v)] = ti;
            ids[static_cast<std::size_t>(v)] = tuple.id;
            const double next_probability = probability * tuple.weight;
            if (depth + 1 == in_.preorder.size()) {
                verify_envelope(next_probability);
                callback(ids, next_probability);
            } else {
                enumerate_depth(depth + 1, ids, chosen, next_probability, callback);
            }
        }
    }

    static bool fixed_size_coin(std::mt19937_64& rng, Count selected,
                                Count remaining) {
        if (selected == 0) return false;
        if (selected > remaining || remaining == 0)
            throw std::logic_error("invalid fixed-size selection state");
        if (selected == remaining) return true;
        std::uniform_int_distribution<Count> distribution(0, remaining - 1);
        return distribution(rng) < selected;
    }

    std::vector<Output> sample_sparse(std::mt19937_64& rng, Count k,
                                      QueryStats& stats) const {
        ++stats.sparse_queries;
        std::unordered_set<Output, VecHash> candidates;
        if (k > static_cast<Count>(std::numeric_limits<std::size_t>::max()))
            throw std::length_error("candidate set exceeds size_t");
        const std::size_t target = static_cast<std::size_t>(k);
        if (target > (std::numeric_limits<std::size_t>::max() - 1) / 2)
            throw std::length_error("candidate reserve overflow");
        candidates.reserve(target * 2 + 1);
        std::vector<Output> output;
        output.reserve(target);
        Output ids(static_cast<std::size_t>(in_.m), -1);
        std::vector<int> chosen(static_cast<std::size_t>(in_.m), -1);
        while (candidates.size() < target) {
            const double probability = uniform_draw_into(rng, ids, chosen);
            ++stats.wr_draws;
            auto [it, inserted] = candidates.insert(ids);
            if (!inserted) {
                ++stats.duplicate_wr_draws;
                continue;
            }
            ++stats.distinct_candidates;
            ++stats.thinning_coins;
            const double keep = probability / envelope_;
            if (!(keep >= 0.0 && keep <= 1.0))
                throw std::runtime_error("invalid sparse thinning probability");
            if (coin(rng, keep)) {
                output.push_back(*it);
                ++stats.outputs;
            }
        }
        return output;
    }

    std::vector<Output> sample_dense(std::mt19937_64& rng, Count k,
                                     QueryStats& stats) const {
        ++stats.dense_queries;
        std::vector<Output> output;
        Count remaining = join_size_;
        Count selected = k;
        enumerate_weighted([&](const Output& ids, double probability) {
            ++stats.enumerated_results;
            const bool take = fixed_size_coin(rng, selected, remaining);
            if (take) {
                --selected;
                ++stats.distinct_candidates;
                ++stats.thinning_coins;
                const double keep = probability / envelope_;
                if (!(keep >= 0.0 && keep <= 1.0))
                    throw std::runtime_error("invalid dense thinning probability");
                if (coin(rng, keep)) {
                    output.push_back(ids);
                    ++stats.outputs;
                }
            }
            --remaining;
        });
        if (remaining != 0 || selected != 0)
            throw std::runtime_error("fixed-size enumeration did not consume K");
        return output;
    }

    void finalize_build_stats() {
        build_stats_.join_size = join_size_;
        build_stats_.mu = mu_;
        build_stats_.max_product_dp = max_probability_dp_;
        build_stats_.envelope = envelope_;
        build_stats_.counts_within_exact_double_integer_range = true;
        build_stats_.double_alias_is_mathematically_exact = false;
        std::size_t bytes = sizeof(*this);
        for (const auto& nd : node_) {
            build_stats_.groups += nd.groups.size();
            bytes += nd.lookup.size() * (sizeof(Key) + sizeof(std::size_t) + 24);
            bytes += nd.groups.capacity() * sizeof(EnvelopeGroup);
            bytes += nd.tuple_completions.capacity() * sizeof(Count);
            bytes += nd.tuple_probability_sum.capacity() * sizeof(double);
            bytes += nd.tuple_max_product.capacity() * sizeof(long double);
            for (const auto& group : nd.groups) {
                build_stats_.alias_entries += group.uniform_alias.item.size();
                bytes += group.tuples.capacity() * sizeof(int);
                bytes += group.uniform_alias.item.capacity() * sizeof(int);
                bytes += group.uniform_alias.prob.capacity() * sizeof(double);
                bytes += group.uniform_alias.alias.capacity() * sizeof(std::size_t);
            }
        }
        bytes += root_blocks_.capacity() * sizeof(RootBlockInfo);
        build_stats_.approximate_bytes = bytes;
    }
};

}
