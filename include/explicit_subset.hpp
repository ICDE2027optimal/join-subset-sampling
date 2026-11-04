// Independent-subset index with dyadic probability buckets and a small-probability tail.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace poisson_join {

class ExplicitSubsetIndex {
public:
    ExplicitSubsetIndex() = default;

    explicit ExplicitSubsetIndex(const std::vector<double>& probabilities) {
        build(probabilities);
    }

    void build(const std::vector<double>& probabilities) {
        clear();
        n_ = probabilities.size();

        const long double side = static_cast<long double>(n_) + 1.0L;
        tail_upper_ = static_cast<double>(1.0L / (side * side));
        if (!(tail_upper_ > 0.0) || tail_upper_ > 1.0)
            throw std::overflow_error("explicit subset tail envelope is not representable");
        tail_log_failure_ = tail_upper_ >= 1.0
            ? -std::numeric_limits<double>::infinity()
            : std::log1p(-tail_upper_);

        std::unordered_map<int, std::size_t> by_upper_exponent;
        by_upper_exponent.reserve(64);

        long double expected = 0.0L;
        for (std::size_t i = 0; i < probabilities.size(); ++i) {
            const double q = probabilities[i];
            if (!std::isfinite(q) || q < 0.0 || q > 1.0)
                throw std::invalid_argument("explicit subset probability must lie in [0,1]");
            expected += static_cast<long double>(q);
            if (q == 0.0) continue;
            if (q == 1.0) {
                certain_.push_back(i);
                continue;
            }
            if (q < tail_upper_) {
                tail_.push_back(Item{i, q});
                continue;
            }

            const int upper_exponent = dyadic_upper_exponent(q);
            auto [it, inserted] = by_upper_exponent.emplace(
                upper_exponent, buckets_.size());
            if (inserted) {
                Bucket bucket;
                bucket.upper = std::ldexp(1.0, upper_exponent);
                if (!(bucket.upper >= q && bucket.upper <= 1.0))
                    throw std::logic_error("invalid dyadic probability envelope");
                buckets_.push_back(std::move(bucket));
            }
            buckets_[it->second].items.push_back(Item{i, q});
        }
        expected_size_ = static_cast<double>(expected);

        std::sort(buckets_.begin(), buckets_.end(),
                  [](const Bucket& a, const Bucket& b) {
                      return a.upper > b.upper;
                  });
        for (Bucket& bucket : buckets_) finalize_bucket(bucket);
        build_first_hit_tables();
    }

    std::vector<std::size_t> query(std::mt19937_64& rng) const {
        std::vector<std::size_t> result;
        if (!certain_.empty()) result.reserve(certain_.size());
        result.insert(result.end(), certain_.begin(), certain_.end());

        std::size_t start = 0;
        while (start < buckets_.size()) {
            const std::size_t next = first_hit_[start].sample(rng);
            if (next == buckets_.size()) break;
            if (next < start || next >= buckets_.size())
                throw std::logic_error("invalid first-hit bucket sample");
            sample_nonempty_bucket(buckets_[next], rng, result);
            start = next + 1;
        }
        sample_tail(rng, result);
        return result;
    }

    std::size_t size() const noexcept { return n_; }
    double expected_size() const noexcept { return expected_size_; }
    std::size_t bucket_count() const noexcept { return buckets_.size(); }
    std::size_t tail_size() const noexcept { return tail_.size(); }
    std::size_t approximate_bytes() const noexcept {
        std::size_t bytes = sizeof(*this);
        bytes += certain_.capacity() * sizeof(std::size_t);
        bytes += tail_.capacity() * sizeof(Item);
        bytes += buckets_.capacity() * sizeof(Bucket);
        for (const Bucket& bucket : buckets_)
            bytes += bucket.items.capacity() * sizeof(Item);
        bytes += first_hit_.capacity() * sizeof(AliasTable);
        for (const AliasTable& table : first_hit_)
            bytes += table.dynamic_bytes();
        return bytes;
    }

private:
    struct Item {
        std::size_t index = 0;
        double probability = 0.0;
    };

    class AliasTable {
    public:
        void build(const std::vector<std::size_t>& items,
                   const std::vector<double>& weights) {
            if (items.size() != weights.size())
                throw std::invalid_argument("alias items/weights size mismatch");
            item_.clear();
            probability_.clear();
            alias_.clear();

            long double total = 0.0L;
            for (std::size_t i = 0; i < items.size(); ++i) {
                const double w = weights[i];
                if (!std::isfinite(w) || w < 0.0)
                    throw std::invalid_argument("invalid alias weight");
                if (w > 0.0) {
                    item_.push_back(items[i]);
                    total += static_cast<long double>(w);
                }
            }
            if (item_.empty() || !(total > 0.0L))
                throw std::invalid_argument("alias table has zero total weight");

            const std::size_t count = item_.size();
            probability_.assign(count, 0.0);
            alias_.assign(count, 0);
            std::vector<long double> scaled(count, 0.0L);
            std::vector<std::size_t> small;
            std::vector<std::size_t> large;
            small.reserve(count);
            large.reserve(count);

            std::size_t k = 0;
            for (std::size_t i = 0; i < items.size(); ++i) {
                if (!(weights[i] > 0.0)) continue;
                scaled[k] = static_cast<long double>(weights[i]) *
                            static_cast<long double>(count) / total;
                (scaled[k] < 1.0L ? small : large).push_back(k);
                ++k;
            }

            while (!small.empty() && !large.empty()) {
                const std::size_t s = small.back();
                small.pop_back();
                const std::size_t l = large.back();
                large.pop_back();
                probability_[s] = static_cast<double>(
                    std::max(0.0L, std::min(1.0L, scaled[s])));
                alias_[s] = l;
                scaled[l] = (scaled[l] + scaled[s]) - 1.0L;
                (scaled[l] < 1.0L ? small : large).push_back(l);
            }
            for (std::size_t i : large) {
                probability_[i] = 1.0;
                alias_[i] = i;
            }
            for (std::size_t i : small) {
                probability_[i] = 1.0;
                alias_[i] = i;
            }
        }

        std::size_t sample(std::mt19937_64& rng) const {
            if (item_.empty()) throw std::logic_error("sample from empty alias table");
            std::uniform_int_distribution<std::size_t> column(0, item_.size() - 1);
            const std::size_t i = column(rng);
            const std::size_t chosen = bernoulli(rng, probability_[i]) ? i : alias_[i];
            return item_[chosen];
        }

        std::size_t dynamic_bytes() const noexcept {
            return item_.capacity() * sizeof(std::size_t) +
                   probability_.capacity() * sizeof(double) +
                   alias_.capacity() * sizeof(std::size_t);
        }

    private:
        std::vector<std::size_t> item_;
        std::vector<double> probability_;
        std::vector<std::size_t> alias_;
    };

    struct Bucket {
        double upper = 0.0;
        double log_failure = 0.0;
        double nonempty_probability = 0.0;
        std::vector<Item> items;
    };

    std::size_t n_ = 0;
    double expected_size_ = 0.0;
    double tail_upper_ = 1.0;
    double tail_log_failure_ = -std::numeric_limits<double>::infinity();
    std::vector<std::size_t> certain_;
    std::vector<Item> tail_;
    std::vector<Bucket> buckets_;
    std::vector<AliasTable> first_hit_;

    void clear() {
        n_ = 0;
        expected_size_ = 0.0;
        tail_upper_ = 1.0;
        tail_log_failure_ = -std::numeric_limits<double>::infinity();
        certain_.clear();
        tail_.clear();
        buckets_.clear();
        first_hit_.clear();
    }

    static int dyadic_upper_exponent(double q) {
        int exponent = 0;
        const double fraction = std::frexp(q, &exponent);

        return fraction == 0.5 ? exponent - 1 : exponent;
    }

    static double uniform_open(std::mt19937_64& rng) {
        
        const std::uint64_t x = rng() >> 12;
        return (static_cast<double>(x) + 0.5) *
               (1.0 / 4503599627370496.0);
    }

    static bool bernoulli(std::mt19937_64& rng, double probability) {
        if (probability <= 0.0) return false;
        if (probability >= 1.0) return true;
        double residual = probability;
        while (true) {
            const double scaled = std::ldexp(residual, 64);
            const std::uint64_t cut = static_cast<std::uint64_t>(scaled);
            const std::uint64_t random = rng();
            if (random < cut) return true;
            if (random > cut) return false;
            residual = scaled - static_cast<double>(cut);
            if (residual == 0.0) return false;
        }
    }

    static double at_least_one(double probability, std::size_t count) {
        if (count == 0 || probability <= 0.0) return 0.0;
        if (probability >= 1.0) return 1.0;
        return -std::expm1(static_cast<double>(count) *
                           std::log1p(-probability));
    }

    void finalize_bucket(Bucket& bucket) {
        if (bucket.items.empty() || !(bucket.upper > 0.0) || bucket.upper > 1.0)
            throw std::logic_error("invalid explicit subset bucket");
        bucket.log_failure = bucket.upper >= 1.0
            ? -std::numeric_limits<double>::infinity()
            : std::log1p(-bucket.upper);
        bucket.nonempty_probability = at_least_one(bucket.upper, bucket.items.size());
        if (!(bucket.nonempty_probability > 0.0) ||
            bucket.nonempty_probability > 1.0)
            throw std::logic_error("invalid bucket nonempty probability");
    }

    void build_first_hit_tables() {
        first_hit_.resize(buckets_.size());
        const std::size_t sentinel = buckets_.size();
        for (std::size_t start = 0; start < buckets_.size(); ++start) {
            std::vector<std::size_t> items;
            std::vector<double> weights;
            items.reserve(buckets_.size() - start + 1);
            weights.reserve(buckets_.size() - start + 1);
            long double survive = 1.0L;
            for (std::size_t j = start; j < buckets_.size(); ++j) {
                const long double q =
                    static_cast<long double>(buckets_[j].nonempty_probability);
                items.push_back(j);
                weights.push_back(static_cast<double>(survive * q));
                survive *= (1.0L - q);
            }
            items.push_back(sentinel);
            weights.push_back(static_cast<double>(survive));
            first_hit_[start].build(items, weights);
        }
    }

    static std::size_t conditional_first_gap(std::mt19937_64& rng,
                                             const Bucket& bucket) {
        if (bucket.upper >= 1.0) return 1;
        const double u = uniform_open(rng);
        const double numerator =
            std::log1p(-u * bucket.nonempty_probability);
        double raw = std::ceil(numerator / bucket.log_failure);
        if (raw < 1.0) raw = 1.0;
        if (static_cast<long double>(raw) >=
            static_cast<long double>(bucket.items.size()))
            return bucket.items.size();
        return static_cast<std::size_t>(raw);
    }

    static bool ordinary_gap(std::mt19937_64& rng, double log_failure,
                             std::size_t remaining, std::size_t& gap) {
        if (remaining == 0) return false;
        if (std::isinf(log_failure) && log_failure < 0.0) {
            gap = 1;
            return true;
        }
        if (!(log_failure < 0.0))
            throw std::logic_error("invalid geometric failure logarithm");
        const double real_gap =
            std::floor(std::log(uniform_open(rng)) / log_failure) + 1.0;
        if (!(real_gap >= 1.0) ||
            static_cast<long double>(real_gap) >
                static_cast<long double>(remaining))
            return false;
        gap = static_cast<std::size_t>(real_gap);
        return gap >= 1 && gap <= remaining;
    }

    static void thin_item(const Item& item, double upper,
                          std::mt19937_64& rng,
                          std::vector<std::size_t>& result) {
        const double keep = item.probability / upper;
        if (!(keep >= 0.0 && keep <= 1.0))
            throw std::logic_error("invalid explicit subset thinning probability");
        if (bernoulli(rng, keep))
            result.push_back(item.index);
    }

    static void sample_nonempty_bucket(const Bucket& bucket,
                                       std::mt19937_64& rng,
                                       std::vector<std::size_t>& result) {
        std::size_t consumed = 0;
        std::size_t gap = conditional_first_gap(rng, bucket);
        while (true) {
            if (gap == 0 || gap > bucket.items.size() - consumed)
                throw std::logic_error("invalid conditional geometric gap");
            const std::size_t position = consumed + gap - 1;
            thin_item(bucket.items[position], bucket.upper, rng, result);
            consumed += gap;
            if (consumed >= bucket.items.size()) break;
            if (!ordinary_gap(rng, bucket.log_failure,
                              bucket.items.size() - consumed, gap))
                break;
        }
    }

    void sample_tail(std::mt19937_64& rng,
                     std::vector<std::size_t>& result) const {
        std::size_t consumed = 0;
        std::size_t gap = 0;
        while (consumed < tail_.size() &&
               ordinary_gap(rng, tail_log_failure_,
                            tail_.size() - consumed, gap)) {
            const std::size_t position = consumed + gap - 1;
            thin_item(tail_[position], tail_upper_, rng, result);
            consumed += gap;
        }
    }
};

}
