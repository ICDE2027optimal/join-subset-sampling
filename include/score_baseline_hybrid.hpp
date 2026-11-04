// PRODUCT score-bucket baseline with exact integer count convolution.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "score_convolution.hpp"

namespace score_baseline_hybrid {

using Count = std::uint64_t;

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

struct ScoreEntry {
    int score = 0;
    Count count = 0;
};
using ScoreVec = std::vector<ScoreEntry>;

inline Count lookup_score(const ScoreVec& a, int score) {
    auto it = std::lower_bound(a.begin(), a.end(), score,
        [](const ScoreEntry& x, int s) { return x.score < s; });
    return it != a.end() && it->score == score ? it->count : 0;
}

struct ScorePrefix {
    int score = 0;
    std::vector<Count> cumulative;
};

struct ScoreGroup {
    Key key = 0;
    std::vector<int> tuples;
    ScoreVec message;
    std::vector<ScorePrefix> prefix;
};

struct ScoreNode {
    std::unordered_map<Key, std::size_t> lookup;
    std::vector<ScoreGroup> groups;
    std::vector<int> tuple_score;
    std::vector<ScoreVec> tuple_total;
    
    std::vector<std::vector<ScoreVec>> suffix;
};

struct FullGroup {
    Key key = 0;
    std::vector<int> tuples;
    std::vector<Count> cumulative;
    Count total = 0;
};

struct FullNode {
    std::unordered_map<Key, std::size_t> lookup;
    std::vector<FullGroup> groups;
    std::vector<Count> tuple_total;
    
    std::vector<std::vector<Count>> child_suffix;
};

struct SampledOutput {
    Output ids;
    double probability = 0.0;
    int score = 0;
};

class CheckedRankIndex {
public:
    explicit CheckedRankIndex(const Instance& in, const std::vector<std::vector<int>>& tuple_score,
                              int score_cap)
        : in_(in), tuple_score_(tuple_score), score_cap_(score_cap), node_(in.m) {
        build();
    }

    Count join_size() const { return join_size_; }

    std::size_t approximate_bytes() const {
        std::size_t bytes = sizeof(*this);
        for (const auto& nd : node_) {
            bytes += nd.lookup.size() * (sizeof(Key) + sizeof(std::size_t) + 24);
            bytes += nd.groups.capacity() * sizeof(FullGroup);
            bytes += nd.tuple_total.capacity() * sizeof(Count);
            bytes += nd.child_suffix.capacity() * sizeof(std::vector<Count>);
            for (const auto& suffix : nd.child_suffix)
                bytes += suffix.capacity() * sizeof(Count);
            for (const auto& group : nd.groups) {
                bytes += group.tuples.capacity() * sizeof(int);
                bytes += group.cumulative.capacity() * sizeof(Count);
            }
        }
        return bytes;
    }

    SampledOutput access(Count rank) const {
        if (rank >= join_size_) throw std::out_of_range("full rank access");
        SampledOutput out;
        out.ids.assign(static_cast<std::size_t>(in_.m), -1);
        out.probability = 1.0;
        out.score = 0;
        access_group(in_.preorder.front(), ROOT_KEY, rank, out);
        return out;
    }

private:
    const Instance& in_;
    const std::vector<std::vector<int>>& tuple_score_;
    int score_cap_;
    std::vector<FullNode> node_;
    Count join_size_ = 0;

    FullGroup* find_group_mut(int v, Key key) {
        auto it = node_[static_cast<std::size_t>(v)].lookup.find(key);
        return it == node_[static_cast<std::size_t>(v)].lookup.end()
            ? nullptr : &node_[static_cast<std::size_t>(v)].groups[it->second];
    }

    const FullGroup* find_group(int v, Key key) const {
        const auto& nd = node_[static_cast<std::size_t>(v)];
        auto it = nd.lookup.find(key);
        return it == nd.lookup.end() ? nullptr : &nd.groups[it->second];
    }

    void build() {
        for (int v = 0; v < in_.m; ++v) {
            auto& nd = node_[static_cast<std::size_t>(v)];
            const std::size_t n = in_.rel[static_cast<std::size_t>(v)].tuples.size();
            nd.tuple_total.assign(n, 0);
            nd.child_suffix.resize(n);
            for (std::size_t ti = 0; ti < n; ++ti) {
                const Tuple& t = in_.rel[static_cast<std::size_t>(v)].tuples[ti];
                const Key key = in_.parent[static_cast<std::size_t>(v)] < 0
                    ? ROOT_KEY : t.edge_key[static_cast<std::size_t>(
                        in_.parent[static_cast<std::size_t>(v)])];
                auto [it, inserted] = nd.lookup.emplace(key, nd.groups.size());
                if (inserted) {
                    nd.groups.push_back(FullGroup{});
                    nd.groups.back().key = key;
                }
                nd.groups[it->second].tuples.push_back(static_cast<int>(ti));
            }
        }

        for (auto order = in_.preorder.rbegin(); order != in_.preorder.rend(); ++order) {
            const int v = *order;
            auto& nd = node_[static_cast<std::size_t>(v)];
            const auto& children = in_.children[static_cast<std::size_t>(v)];
            for (std::size_t ti = 0; ti < in_.rel[static_cast<std::size_t>(v)].tuples.size(); ++ti) {
                const Tuple& t = in_.rel[static_cast<std::size_t>(v)].tuples[ti];
                auto& suffix = nd.child_suffix[ti];
                suffix.assign(children.size() + 1, 1);
                for (std::size_t h = children.size(); h-- > 0;) {
                    const int child = children[h];
                    const FullGroup* cg =
                        find_group(child, t.edge_key[static_cast<std::size_t>(child)]);
                    const Count child_total = cg ? cg->total : 0;
                    suffix[h] = checked_mul(child_total, suffix[h + 1],
                                            "full-rank child suffix");
                }
                nd.tuple_total[ti] = suffix[0];
            }
            for (auto& group : nd.groups) {
                group.cumulative.clear();
                group.cumulative.reserve(group.tuples.size());
                group.total = 0;
                for (int ti : group.tuples) {
                    group.total = checked_add(
                        group.total, nd.tuple_total[static_cast<std::size_t>(ti)],
                        "full-rank group prefix");
                    group.cumulative.push_back(group.total);
                }
            }
        }

        const FullGroup* root = find_group(in_.preorder.front(), ROOT_KEY);
        join_size_ = root ? root->total : 0;
    }

    void access_group(int v, Key key, Count rank, SampledOutput& out) const {
        const FullGroup* group = find_group(v, key);
        if (!group || rank >= group->total)
            throw std::runtime_error("invalid full-rank group access");
        auto pos_it = std::upper_bound(group->cumulative.begin(), group->cumulative.end(), rank);
        const std::size_t pos = static_cast<std::size_t>(pos_it - group->cumulative.begin());
        const Count before = pos == 0 ? 0 : group->cumulative[pos - 1];
        Count local = rank - before;
        const int ti = group->tuples[pos];
        const Tuple& t = in_.rel[static_cast<std::size_t>(v)].tuples[static_cast<std::size_t>(ti)];
        out.ids[static_cast<std::size_t>(v)] = t.id;
        out.probability *= t.weight;
        out.score = std::min(score_cap_, out.score +
            tuple_score_[static_cast<std::size_t>(v)][static_cast<std::size_t>(ti)]);

        const auto& children = in_.children[static_cast<std::size_t>(v)];
        const auto& suffix = node_[static_cast<std::size_t>(v)].child_suffix[
            static_cast<std::size_t>(ti)];
        for (std::size_t h = 0; h < children.size(); ++h) {
            const Count later = suffix[h + 1];
            const int child = children[h];
            const FullGroup* cg =
                find_group(child, t.edge_key[static_cast<std::size_t>(child)]);
            if (!cg || cg->total == 0 || later == 0)
                throw std::runtime_error("zero child reached by positive full rank");
            const Count child_rank = local / later;
            local %= later;
            if (child_rank >= cg->total)
                throw std::runtime_error("full-rank Cartesian decomposition failed");
            access_group(child, t.edge_key[static_cast<std::size_t>(child)],
                         child_rank, out);
        }
    }
};

struct Config {
    double h = 1.0;
    int L_override = -1;

    double rho_upper = 0.0;
};

struct QueryStats {
    Count active_head_buckets = 0;
    Count head_candidates = 0;
    Count head_outputs = 0;
    Count tail_rank_probes = 0;
    Count tail_outputs = 0;
};

struct BuildStats {
    std::size_t input_tuples = 0;
    int score_limit = 0;
    double score_quantum = 0.0;
    Count join_size = 0;
    Count convolution_products = 0;
    std::size_t tuple_score_entries = 0;
    std::size_t tuple_total_score_entries = 0;
    std::size_t suffix_score_entries = 0;
    std::size_t group_message_entries = 0;
    std::size_t prefix_cells = 0;
    std::size_t meta_buckets = 0;
    std::size_t approximate_bytes = 0;
    std::size_t temporary_count_cells_peak = 0;
    score_convolution::Stats convolution;
};

class ScoreBucketIndex {
public:
    explicit ScoreBucketIndex(const Instance& in, Config config = {})
        : in_(in), config_(config), node_(static_cast<std::size_t>(in.m)) {
        if (in_.m <= 0 || in_.preorder.empty())
            throw std::invalid_argument("score baseline needs a nonempty join tree");
        if (!(config_.h > 0.0) || !std::isfinite(config_.h))
            throw std::invalid_argument("score quantum h must be finite and positive");
        std::size_t n = 0;
        for (const auto& rel : in_.rel) n += rel.tuples.size();
        const double rho = config_.rho_upper > 0.0 ? config_.rho_upper
                                                   : static_cast<double>(in_.m);
        if (config_.L_override >= 1) {
            L_ = config_.L_override;
        } else {
            const double safe_n = static_cast<double>(std::max<std::size_t>(n, 2));
            L_ = static_cast<int>(std::ceil(2.0 * rho * std::log2(safe_n) / config_.h));
            L_ = std::max(L_, 1);
        }
        build_scores_and_groups();
        build_score_messages();
        full_rank_ = std::make_unique<CheckedRankIndex>(in_, tuple_score_cache_, L_);
        build_meta_sampler();
        finalize_build_stats(n);
    }

    explicit ScoreBucketIndex(const Instance& in, double score_quantum)
        : ScoreBucketIndex(in, Config{score_quantum, -1, 0.0}) {}

    ScoreBucketIndex(const ScoreBucketIndex&) = delete;
    ScoreBucketIndex& operator=(const ScoreBucketIndex&) = delete;
    ScoreBucketIndex(ScoreBucketIndex&&) = delete;
    ScoreBucketIndex& operator=(ScoreBucketIndex&&) = delete;

    int L() const { return L_; }
    double h() const { return config_.h; }
    Count join_size() const { return full_rank_->join_size(); }
    const BuildStats& build_stats() const { return build_stats_; }
    std::size_t approximate_bytes() const { return build_stats_.approximate_bytes; }

    int score_of_weight(double p) const {
        if (!(p >= 0.0 && p <= 1.0) || std::isnan(p))
            throw std::invalid_argument("tuple weight must lie in [0,1]");
        if (p == 0.0) return L_;
        const long double h = static_cast<long double>(config_.h);
        const long double value = -std::log2(static_cast<long double>(p)) / h;
        long long s = static_cast<long long>(std::floor(value));
        if (s < 0) s = 0;

        while (s > 0 &&
               static_cast<long double>(p) > std::exp2(-h * static_cast<long double>(s)))
            --s;
        while (s < L_ &&
               static_cast<long double>(p) <=
                   std::exp2(-h * static_cast<long double>(s + 1)))
            ++s;
        return static_cast<int>(std::min<long long>(s, L_));
    }

    Count bucket_count(int score) const {
        if (score < 0 || score >= L_) return 0;
        const ScoreGroup* root = find_group(in_.preorder.front(), ROOT_KEY);
        return root ? lookup_score(root->message, score) : 0;
    }

    double bucket_upper(int score) const {
        if (score < 0) throw std::invalid_argument("negative score");
        return std::exp2(-config_.h * static_cast<double>(score));
    }

    SampledOutput access_score(int score, Count rank) const {
        const Count count = bucket_count(score);
        if (rank >= count) throw std::out_of_range("score-bucket rank access");
        SampledOutput out;
        out.ids.assign(static_cast<std::size_t>(in_.m), -1);
        out.probability = 1.0;
        out.score = 0;
        access_score_group(in_.preorder.front(), ROOT_KEY, score, rank, out);
        if (out.score != score)
            throw std::runtime_error("score rank access returned wrong score");
        return out;
    }

    SampledOutput access_full(Count rank) const { return full_rank_->access(rank); }

    std::vector<Output>
    sample_vector(std::mt19937_64& rng, QueryStats& stats) const {
        std::vector<Output> result;
        sample_head(rng, stats, result);
        sample_tail(rng, stats, result);
        return result;
    }

    std::unordered_set<Output, VecHash>
    sample_set(std::mt19937_64& rng, QueryStats& stats) const {
        auto vector = sample_vector(rng, stats);
        return std::unordered_set<Output, VecHash>(
            std::make_move_iterator(vector.begin()),
            std::make_move_iterator(vector.end()));
    }

private:
    struct MetaBucket {
        int score = 0;
        Count count = 0;
        double upper = 0.0;
        double nonempty_probability = 0.0;
        double log_failure = 0.0;
    };

    const Instance& in_;
    Config config_;
    int L_ = 0;
    std::vector<ScoreNode> node_;
    std::vector<std::vector<int>> tuple_score_cache_;
    std::unique_ptr<CheckedRankIndex> full_rank_;
    std::vector<MetaBucket> meta_;
    std::vector<AliasTable> first_hit_;
    BuildStats build_stats_;

    ScoreGroup* find_group_mut(int v, Key key) {
        auto& nd = node_[static_cast<std::size_t>(v)];
        auto it = nd.lookup.find(key);
        return it == nd.lookup.end() ? nullptr : &nd.groups[it->second];
    }

    const ScoreGroup* find_group(int v, Key key) const {
        const auto& nd = node_[static_cast<std::size_t>(v)];
        auto it = nd.lookup.find(key);
        return it == nd.lookup.end() ? nullptr : &nd.groups[it->second];
    }

    static const ScorePrefix* find_prefix(const ScoreGroup& group, int score) {
        auto it = std::lower_bound(group.prefix.begin(), group.prefix.end(), score,
            [](const ScorePrefix& p, int s) { return p.score < s; });
        return it != group.prefix.end() && it->score == score ? &*it : nullptr;
    }

    void build_scores_and_groups() {
        tuple_score_cache_.resize(static_cast<std::size_t>(in_.m));
        for (int v = 0; v < in_.m; ++v) {
            auto& nd = node_[static_cast<std::size_t>(v)];
            const auto& tuples = in_.rel[static_cast<std::size_t>(v)].tuples;
            nd.tuple_score.resize(tuples.size());
            nd.tuple_total.resize(tuples.size());
            nd.suffix.resize(tuples.size());
            for (std::size_t ti = 0; ti < tuples.size(); ++ti) {
                const Tuple& t = tuples[ti];
                nd.tuple_score[ti] = score_of_weight(t.weight);
                const Key key = in_.parent[static_cast<std::size_t>(v)] < 0
                    ? ROOT_KEY : t.edge_key[static_cast<std::size_t>(
                        in_.parent[static_cast<std::size_t>(v)])];
                auto [it, inserted] = nd.lookup.emplace(key, nd.groups.size());
                if (inserted) {
                    nd.groups.push_back(ScoreGroup{});
                    nd.groups.back().key = key;
                }
                nd.groups[it->second].tuples.push_back(static_cast<int>(ti));
            }
            tuple_score_cache_[static_cast<std::size_t>(v)] = nd.tuple_score;
        }
    }

    void build_score_messages() {
        const ScoreVec unit{{0, 1}};
        for (auto order = in_.preorder.rbegin(); order != in_.preorder.rend(); ++order) {
            const int v = *order;
            auto& nd = node_[static_cast<std::size_t>(v)];
            const auto& children = in_.children[static_cast<std::size_t>(v)];
            for (std::size_t ti = 0; ti < in_.rel[static_cast<std::size_t>(v)].tuples.size(); ++ti) {
                const Tuple& t = in_.rel[static_cast<std::size_t>(v)].tuples[ti];
                auto& suffix = nd.suffix[ti];
                suffix.resize(children.size() + 1);
                suffix[children.size()] = unit;
                for (std::size_t h = children.size(); h-- > 0;) {
                    const int child = children[h];
                    const ScoreGroup* cg =
                        find_group(child, t.edge_key[static_cast<std::size_t>(child)]);
                    suffix[h] = cg ? score_convolution::convolve(
                                         cg->message, suffix[h + 1], L_,
                                         &build_stats_.convolution)
                                   : ScoreVec{};
                }
                const int own = nd.tuple_score[ti];
                if (own >= L_ || suffix[0].empty()) continue;
                auto& total = nd.tuple_total[ti];
                total.reserve(suffix[0].size());
                for (const auto& x : suffix[0]) {
                    if (own + x.score < L_)
                        total.push_back({own + x.score, x.count});
                }
            }

            for (auto& group : nd.groups) {
                std::vector<Count> aggregate(static_cast<std::size_t>(L_), 0);
                for (int ti : group.tuples) {
                    for (const auto& x : nd.tuple_total[static_cast<std::size_t>(ti)]) {
                        auto& dst = aggregate[static_cast<std::size_t>(x.score)];
                        dst = checked_add(dst, x.count, "score group aggregation");
                    }
                }
                for (int score = 0; score < L_; ++score) {
                    const Count total = aggregate[static_cast<std::size_t>(score)];
                    if (total == 0) continue;
                    group.message.push_back({score, total});
                    ScorePrefix prefix;
                    prefix.score = score;
                    prefix.cumulative.reserve(group.tuples.size());
                    Count running = 0;
                    for (int ti : group.tuples) {
                        running = checked_add(
                            running,
                            lookup_score(nd.tuple_total[static_cast<std::size_t>(ti)], score),
                            "score tuple prefix");
                        prefix.cumulative.push_back(running);
                    }
                    if (running != total)
                        throw std::logic_error("score prefix total mismatch");
                    group.prefix.push_back(std::move(prefix));
                }
            }
        }
    }

    void build_meta_sampler() {
        const ScoreGroup* root = find_group(in_.preorder.front(), ROOT_KEY);
        if (root) {
            for (const auto& x : root->message) {
                const double upper = bucket_upper(x.score);
                const double q = upper >= 1.0
                    ? 1.0
                    : -std::expm1(static_cast<double>(x.count) * std::log1p(-upper));
                if (!(q >= 0.0 && q <= 1.0))
                    throw std::runtime_error("invalid score meta probability");
                if (q > 0.0) {
                    const double log_failure =
                        upper >= 1.0 ? -std::numeric_limits<double>::infinity()
                                     : std::log1p(-upper);
                    meta_.push_back({x.score, x.count, upper, q, log_failure});
                }
            }
        }
        first_hit_.resize(meta_.size());
        for (std::size_t start = 0; start < meta_.size(); ++start) {
            std::vector<int> item;
            std::vector<double> weight;
            double survive = 1.0;
            for (std::size_t j = start; j < meta_.size(); ++j) {
                const double q = meta_[j].nonempty_probability;
                item.push_back(static_cast<int>(j));
                weight.push_back(survive * q);
                survive *= (1.0 - q);
            }
            item.push_back(-1);
            weight.push_back(survive);
            first_hit_[start].build(item, weight);
            if (first_hit_[start].empty())
                throw std::runtime_error("empty first-hit alias table");
        }
    }

    void finalize_build_stats(std::size_t input_tuples) {
        build_stats_.input_tuples = input_tuples;
        build_stats_.score_limit = L_;
        build_stats_.score_quantum = config_.h;
        build_stats_.join_size = full_rank_->join_size();
        build_stats_.meta_buckets = meta_.size();
        std::size_t bytes = sizeof(*this) + full_rank_->approximate_bytes();
        for (const auto& nd : node_) {
            build_stats_.tuple_score_entries += nd.tuple_score.size();
            bytes += nd.lookup.size() * (sizeof(Key) + sizeof(std::size_t) + 24);
            bytes += nd.groups.capacity() * sizeof(ScoreGroup);
            bytes += nd.tuple_score.capacity() * sizeof(int);
            bytes += nd.tuple_total.capacity() * sizeof(ScoreVec);
            bytes += nd.suffix.capacity() * sizeof(std::vector<ScoreVec>);
            for (const auto& total : nd.tuple_total) {
                build_stats_.tuple_total_score_entries += total.size();
                bytes += total.capacity() * sizeof(ScoreEntry);
            }
            for (const auto& tuple_suffix : nd.suffix) {
                bytes += tuple_suffix.capacity() * sizeof(ScoreVec);
                for (const auto& suffix : tuple_suffix) {
                    build_stats_.suffix_score_entries += suffix.size();
                    bytes += suffix.capacity() * sizeof(ScoreEntry);
                }
            }
            for (const auto& group : nd.groups) {
                build_stats_.group_message_entries += group.message.size();
                bytes += group.tuples.capacity() * sizeof(int);
                bytes += group.message.capacity() * sizeof(ScoreEntry);
                bytes += group.prefix.capacity() * sizeof(ScorePrefix);
                for (const auto& prefix : group.prefix) {
                    build_stats_.prefix_cells += prefix.cumulative.size();
                    bytes += prefix.cumulative.capacity() * sizeof(Count);
                }
            }
        }
        bytes += tuple_score_cache_.capacity() * sizeof(std::vector<int>);
        for (const auto& scores : tuple_score_cache_)
            bytes += scores.capacity() * sizeof(int);
        bytes += meta_.capacity() * sizeof(MetaBucket);
        bytes += first_hit_.capacity() * sizeof(AliasTable);
        for (const auto& table : first_hit_) {
            bytes += table.item.capacity() * sizeof(int);
            bytes += table.prob.capacity() * sizeof(double);
            bytes += table.alias.capacity() * sizeof(std::size_t);
        }
        build_stats_.approximate_bytes = bytes;
        build_stats_.temporary_count_cells_peak = static_cast<std::size_t>(L_);
        build_stats_.convolution_products =
            build_stats_.convolution.direct_pairs;
    }

    static double at_least_one(double p, Count n) {
        if (n == 0 || p <= 0.0) return 0.0;
        if (p >= 1.0) return 1.0;
        return -std::expm1(static_cast<double>(n) * std::log1p(-p));
    }

    static Count truncated_first(std::mt19937_64& rng, double p, Count n,
                                 double nonempty_probability) {
        if (n == 0 || !(nonempty_probability > 0.0))
            throw std::invalid_argument("empty truncated geometric");
        if (p >= 1.0) return 1;
        const double u = uniform_open(rng);
        const double numerator = std::log1p(-u * nonempty_probability);
        const double denominator = std::log1p(-p);
        double raw = std::ceil(numerator / denominator);
        if (raw < 1.0) raw = 1.0;
        if (static_cast<long double>(raw) >= static_cast<long double>(n))
            return n;
        return static_cast<Count>(raw);
    }

    static bool next_success(std::mt19937_64& rng, double p, Count remaining,
                             Count& gap) {
        const double q = at_least_one(p, remaining);
        if (!coin(rng, q)) return false;
        gap = truncated_first(rng, p, remaining, q);
        return true;
    }

    static bool ordinary_geometric_gap(std::mt19937_64& rng,
                                       double log_failure, Count remaining,
                                       Count& gap) {
        if (remaining == 0) return false;
        if (std::isinf(log_failure) && log_failure < 0.0) {
            gap = 1;
            return true;
        }
        if (!(log_failure < 0.0))
            throw std::invalid_argument("ordinary geometric needs log(1-p)<0");
        const double real_gap =
            std::floor(std::log(uniform_open(rng)) / log_failure) + 1.0;

        if (!(real_gap >= 1.0) ||
            static_cast<long double>(real_gap) >
                static_cast<long double>(remaining))
            return false;
        gap = static_cast<Count>(real_gap);
        return gap >= 1 && gap <= remaining;
    }

    void access_score_group(int v, Key key, int score, Count rank,
                            SampledOutput& out) const {
        const ScoreGroup* group = find_group(v, key);
        if (!group) throw std::runtime_error("missing score group");
        const ScorePrefix* prefix = find_prefix(*group, score);
        if (!prefix || prefix->cumulative.empty() || rank >= prefix->cumulative.back())
            throw std::runtime_error("invalid score-group rank");
        auto pos_it = std::upper_bound(prefix->cumulative.begin(),
                                       prefix->cumulative.end(), rank);
        const std::size_t pos =
            static_cast<std::size_t>(pos_it - prefix->cumulative.begin());
        const Count before = pos == 0 ? 0 : prefix->cumulative[pos - 1];
        const Count local = rank - before;
        const int ti = group->tuples[pos];
        const Tuple& t = in_.rel[static_cast<std::size_t>(v)].tuples[
            static_cast<std::size_t>(ti)];
        const int own = node_[static_cast<std::size_t>(v)].tuple_score[
            static_cast<std::size_t>(ti)];
        if (own > score) throw std::logic_error("negative residual score");
        out.ids[static_cast<std::size_t>(v)] = t.id;
        out.probability *= t.weight;
        out.score += own;
        access_suffix(v, ti, 0, score - own, local, out);
    }

    void access_suffix(int v, int ti, std::size_t h, int residual, Count rank,
                       SampledOutput& out) const {
        const auto& children = in_.children[static_cast<std::size_t>(v)];
        if (h == children.size()) {
            if (residual != 0 || rank != 0)
                throw std::runtime_error("invalid terminal suffix rank");
            return;
        }
        const Tuple& t = in_.rel[static_cast<std::size_t>(v)].tuples[
            static_cast<std::size_t>(ti)];
        const int child = children[h];
        const ScoreGroup* child_group =
            find_group(child, t.edge_key[static_cast<std::size_t>(child)]);
        if (!child_group) throw std::runtime_error("missing positive child group");
        const ScoreVec& rest = node_[static_cast<std::size_t>(v)].suffix[
            static_cast<std::size_t>(ti)][h + 1];
        std::size_t rest_pos = rest.size();
        while (rest_pos > 0 && rest[rest_pos - 1].score > residual) --rest_pos;
        for (const auto& child_part : child_group->message) {
            if (child_part.score > residual) break;
            const int needed = residual - child_part.score;
            while (rest_pos > 0 && rest[rest_pos - 1].score > needed) --rest_pos;
            const Count rest_count =
                rest_pos > 0 && rest[rest_pos - 1].score == needed
                    ? rest[rest_pos - 1].count : 0;
            if (rest_count == 0) continue;
            const Count block = checked_mul(child_part.count, rest_count,
                                            "score access Cartesian block");
            if (rank >= block) {
                rank -= block;
                continue;
            }
            const Count child_rank = rank / rest_count;
            const Count rest_rank = rank % rest_count;
            access_score_group(child, t.edge_key[static_cast<std::size_t>(child)],
                               child_part.score, child_rank, out);
            access_suffix(v, ti, h + 1, residual - child_part.score,
                          rest_rank, out);
            return;
        }
        throw std::runtime_error("score suffix rank not covered");
    }

    void sample_bucket(const MetaBucket& bucket, std::mt19937_64& rng,
                       QueryStats& stats,
                       std::vector<Output>& result) const {
        Count consumed = 0;
        Count gap = truncated_first(rng, bucket.upper, bucket.count,
                                    bucket.nonempty_probability);
        while (true) {
            if (gap == 0 || gap > bucket.count - consumed)
                throw std::runtime_error("invalid bucket geometric gap");
            const Count rank = consumed + gap - 1;
            SampledOutput candidate = access_score(bucket.score, rank);
            ++stats.head_candidates;
            const double keep = candidate.probability / bucket.upper;
            if (!(keep >= 0.0 && keep <= 1.0 + 1e-12))
                throw std::runtime_error("invalid head thinning probability");
            if (coin(rng, std::min(1.0, keep))) {
                result.push_back(std::move(candidate.ids));
                ++stats.head_outputs;
            }
            consumed += gap;
            if (consumed >= bucket.count) break;
            if (!ordinary_geometric_gap(
                    rng, bucket.log_failure, bucket.count - consumed, gap))
                break;
        }
    }

    void sample_head(std::mt19937_64& rng, QueryStats& stats,
                     std::vector<Output>& result) const {
        std::size_t start = 0;
        while (start < meta_.size()) {
            const int next = first_hit_[start].sample(rng);
            if (next < 0) break;
            const std::size_t j = static_cast<std::size_t>(next);
            if (j < start || j >= meta_.size())
                throw std::runtime_error("invalid first-hit meta sample");
            ++stats.active_head_buckets;
            sample_bucket(meta_[j], rng, stats, result);
            start = j + 1;
        }
    }

    void sample_tail(std::mt19937_64& rng, QueryStats& stats,
                     std::vector<Output>& result) const {
        const Count total = full_rank_->join_size();
        if (total == 0) return;
        const double upper = bucket_upper(L_);
        if (!(upper > 0.0))
            throw std::underflow_error("tail upper probability underflowed double");
        Count consumed = 0;
        Count gap = 0;
        while (consumed < total &&
               next_success(rng, upper, total - consumed, gap)) {
            const Count rank = consumed + gap - 1;
            SampledOutput candidate = full_rank_->access(rank);
            ++stats.tail_rank_probes;
            if (candidate.score >= L_) {
                const double keep = candidate.probability / upper;
                if (!(keep >= 0.0 && keep <= 1.0 + 1e-12))
                    throw std::runtime_error("invalid tail thinning probability");
                if (coin(rng, std::min(1.0, keep))) {
                    result.push_back(std::move(candidate.ids));
                    ++stats.tail_outputs;
                }
            }
            consumed += gap;
        }
    }
};

}
