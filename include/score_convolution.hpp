// Construction-time sparse, direct, and NTT convolution with checked CRT.
#pragma once

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <atcoder/convolution>

namespace score_convolution {

using Count = std::uint64_t;

inline constexpr std::uint32_t PRIME1 = 754974721U;
inline constexpr std::uint32_t PRIME2 = 167772161U;
inline constexpr std::uint32_t PRIME3 = 469762049U;
inline constexpr Count PRIME12 =
    static_cast<Count>(PRIME1) * static_cast<Count>(PRIME2);
inline constexpr std::size_t MAX_COMMON_TRANSFORM = std::size_t{1} << 24;

enum class DispatchMode {
    automatic,
    force_direct,
    force_ntt,
};

struct Config {
    DispatchMode mode = DispatchMode::automatic;
    Count fixed_direct_pair_limit = 4096;
    Count transform_cost_factor = 8;
};

struct Stats {
    Count calls = 0;
    Count empty_calls = 0;
    Count singleton_calls = 0;
    Count direct_calls = 0;
    Count ntt2_calls = 0;
    Count ntt3_calls = 0;
    Count direct_pairs = 0;
    Count transform_points_times_primes = 0;
    Count crt_coefficients = 0;
    Count trimmed_entries = 0;
    std::size_t max_transform_length = 0;
};

inline Count checked_add(Count a, Count b, const char* where) {
    if (std::numeric_limits<Count>::max() - a < b)
        throw std::overflow_error(std::string("uint64 overflow at ") + where);
    return a + b;
}

inline Count checked_mul(Count a, Count b, const char* where) {
    if (a != 0 && b > std::numeric_limits<Count>::max() / a)
        throw std::overflow_error(std::string("uint64 overflow at ") + where);
    return a * b;
}

inline void add_stat(Count& dst, Count value, const char* where) {
    dst = checked_add(dst, value, where);
}

inline Count pow_mod(Count base, Count exponent, Count modulus) {
    Count result = 1;
    while (exponent != 0) {
        if (exponent & 1U)
            result = static_cast<Count>(
                (static_cast<unsigned __int128>(result) * base) % modulus);
        base = static_cast<Count>(
            (static_cast<unsigned __int128>(base) * base) % modulus);
        exponent >>= 1U;
    }
    return result;
}

inline Count inverse_prime(Count value, Count prime) {
    if (value == 0 || value >= prime)
        throw std::invalid_argument("invalid modular inverse operand");
    return pow_mod(value, prime - 2, prime);
}

inline std::size_t checked_bit_ceil(std::size_t value) {
    if (value == 0) return 1;
    if (value > (std::numeric_limits<std::size_t>::max() >> 1) + 1)
        throw std::overflow_error("convolution length cannot be rounded to a power of two");
    return std::bit_ceil(value);
}

template <class ScoreVec>
void validate_input(const ScoreVec& input, const char* name) {
    int previous = -1;
    for (const auto& entry : input) {
        if (entry.score < 0 || entry.score <= previous)
            throw std::invalid_argument(std::string(name) +
                                        " scores must be nonnegative and strictly increasing");
        if (entry.count == 0)
            throw std::invalid_argument(std::string(name) +
                                        " must not store zero coefficients");
        previous = entry.score;
    }
}

template <class ScoreVec>
Count mass_prefix(const ScoreVec& input, std::size_t end, const char* where) {
    Count total = 0;
    for (std::size_t i = 0; i < end; ++i)
        total = checked_add(total, static_cast<Count>(input[i].count), where);
    return total;
}

template <class ScoreVec>
Count feasible_pairs(const ScoreVec& a, std::size_t a_end,
                     const ScoreVec& b, std::size_t b_end, int limit) {
    std::size_t b_prefix = b_end;
    unsigned __int128 total = 0;
    for (std::size_t i = 0; i < a_end; ++i) {
        while (b_prefix != 0 &&
               static_cast<std::int64_t>(a[i].score) +
                       static_cast<std::int64_t>(b[b_prefix - 1].score) >=
                   static_cast<std::int64_t>(limit))
            --b_prefix;
        total += b_prefix;
    }
    if (total > std::numeric_limits<Count>::max())
        throw std::overflow_error("score convolution support-pair count overflow");
    return static_cast<Count>(total);
}

template <class ScoreVec>
ScoreVec direct_convolution(const ScoreVec& a, std::size_t a_end,
                            const ScoreVec& b, std::size_t b_end, int limit,
                            Count coefficient_bound) {
    const int offset = a.front().score + b.front().score;

    const std::int64_t largest_sum =
        static_cast<std::int64_t>(a[a_end - 1].score) +
        static_cast<std::int64_t>(b[b_end - 1].score);
    const int maximum = static_cast<int>(std::min<std::int64_t>(
        static_cast<std::int64_t>(limit) - 1, largest_sum));
    std::vector<Count> dense(static_cast<std::size_t>(maximum - offset + 1), 0);
    for (std::size_t i = 0; i < a_end; ++i) {
        for (std::size_t j = 0; j < b_end; ++j) {
            const std::int64_t score = static_cast<std::int64_t>(a[i].score) +
                                       static_cast<std::int64_t>(b[j].score);
            if (score >= limit) break;
            const Count product = checked_mul(
                static_cast<Count>(a[i].count), static_cast<Count>(b[j].count),
                "direct score convolution product");
            Count& destination = dense[static_cast<std::size_t>(score - offset)];
            destination = checked_add(destination, product,
                                      "direct score convolution accumulation");
            if (destination > coefficient_bound)
                throw std::logic_error("direct coefficient exceeds mass certificate");
        }
    }
    ScoreVec output;
    for (std::size_t i = 0; i < dense.size(); ++i)
        if (dense[i] != 0)
            output.push_back({offset + static_cast<int>(i), dense[i]});
    return output;
}

template <std::uint32_t Modulus, class ScoreVec>
std::vector<std::uint32_t> residue_convolution(
    const ScoreVec& a, std::size_t a_end, int a_offset, std::size_t a_span,
    const ScoreVec& b, std::size_t b_end, int b_offset, std::size_t b_span) {
    std::vector<std::uint32_t> dense_a(a_span, 0);
    std::vector<std::uint32_t> dense_b(b_span, 0);
    for (std::size_t i = 0; i < a_end; ++i)
        dense_a[static_cast<std::size_t>(a[i].score - a_offset)] =
            static_cast<std::uint32_t>(static_cast<Count>(a[i].count) % Modulus);
    for (std::size_t i = 0; i < b_end; ++i)
        dense_b[static_cast<std::size_t>(b[i].score - b_offset)] =
            static_cast<std::uint32_t>(static_cast<Count>(b[i].count) % Modulus);
    return atcoder::convolution<Modulus>(dense_a, dense_b);
}

inline Count reconstruct_two(std::uint32_t r1, std::uint32_t r2) {
    static const Count inverse_p1_mod_p2 =
        inverse_prime(static_cast<Count>(PRIME1) % PRIME2, PRIME2);
    const Count r1_mod_p2 = static_cast<Count>(r1) % PRIME2;
    const Count delta =
        (static_cast<Count>(r2) + PRIME2 - r1_mod_p2) % PRIME2;
    const Count digit = static_cast<Count>(
        (static_cast<unsigned __int128>(delta) * inverse_p1_mod_p2) % PRIME2);
    return static_cast<Count>(r1) + static_cast<Count>(PRIME1) * digit;
}

inline unsigned __int128 reconstruct_three(std::uint32_t r1,
                                           std::uint32_t r2,
                                           std::uint32_t r3) {
    static const Count p12_mod_p3 = static_cast<Count>(PRIME12 % PRIME3);
    static const Count inverse_p12_mod_p3 = inverse_prime(p12_mod_p3, PRIME3);
    const Count first = reconstruct_two(r1, r2);
    const Count first_mod_p3 = first % PRIME3;
    const Count delta =
        (static_cast<Count>(r3) + PRIME3 - first_mod_p3) % PRIME3;
    const Count digit = static_cast<Count>(
        (static_cast<unsigned __int128>(delta) * inverse_p12_mod_p3) % PRIME3);
    return static_cast<unsigned __int128>(first) +
           static_cast<unsigned __int128>(PRIME12) * digit;
}

template <class ScoreVec>
void validate_output_mass(const ScoreVec& output, Count coefficient_bound,
                          bool untruncated) {
    unsigned __int128 total = 0;
    for (const auto& entry : output) {
        if (static_cast<Count>(entry.count) > coefficient_bound)
            throw std::logic_error("score coefficient exceeds mass certificate");
        total += static_cast<Count>(entry.count);
    }
    if (total > coefficient_bound)
        throw std::logic_error("score convolution output mass exceeds certificate");
    if (untruncated && total != coefficient_bound)
        throw std::logic_error("untruncated score convolution lost mass");
}

template <class ScoreVec>
ScoreVec convolve(const ScoreVec& a, const ScoreVec& b, int limit,
                  Stats* stats = nullptr, Config config = {}) {
    if (limit <= 0) throw std::invalid_argument("score convolution needs limit > 0");
    if (config.transform_cost_factor == 0)
        throw std::invalid_argument("transform cost factor must be positive");
    validate_input(a, "left score vector");
    validate_input(b, "right score vector");
    Stats local;
    Stats& accounting = stats ? *stats : local;
    add_stat(accounting.calls, 1, "convolution call counter");
    if (a.empty() || b.empty()) {
        add_stat(accounting.empty_calls, 1, "empty convolution call counter");
        return {};
    }

    const std::int64_t minimum_sum = static_cast<std::int64_t>(a.front().score) +
                                     static_cast<std::int64_t>(b.front().score);
    if (minimum_sum >= limit) {
        add_stat(accounting.empty_calls, 1, "empty convolution call counter");
        add_stat(accounting.trimmed_entries,
                 static_cast<Count>(a.size() + b.size()),
                 "trimmed score-entry counter");
        return {};
    }

    std::size_t a_end = 0;
    while (a_end < a.size() &&
           static_cast<std::int64_t>(a[a_end].score) + b.front().score < limit)
        ++a_end;
    std::size_t b_end = 0;
    while (b_end < b.size() &&
           static_cast<std::int64_t>(b[b_end].score) + a.front().score < limit)
        ++b_end;
    if (a_end == 0 || b_end == 0)
        throw std::logic_error("positive minimum unexpectedly trimmed away");
    add_stat(accounting.trimmed_entries,
             static_cast<Count>((a.size() - a_end) + (b.size() - b_end)),
             "trimmed score-entry counter");

    const Count mass_a = mass_prefix(a, a_end, "left score mass");
    const Count mass_b = mass_prefix(b, b_end, "right score mass");

    const Count coefficient_bound =
        checked_mul(mass_a, mass_b, "score convolution mass certificate");
    const Count pairs = feasible_pairs(a, a_end, b, b_end, limit);
    if (pairs == 0) throw std::logic_error("positive score convolution has no pairs");

    const int a_offset = a.front().score;
    const int b_offset = b.front().score;
    const std::size_t a_span = static_cast<std::size_t>(
        a[a_end - 1].score - a_offset + 1);
    const std::size_t b_span = static_cast<std::size_t>(
        b[b_end - 1].score - b_offset + 1);
    if (a_span > std::numeric_limits<std::size_t>::max() - b_span + 1)
        throw std::overflow_error("score convolution span overflow");
    const std::size_t full_length = a_span + b_span - 1;
    const std::size_t transform_length = checked_bit_ceil(full_length);
    const Count prime_count = coefficient_bound < PRIME12 ? 2 : 3;

    bool use_direct = config.mode == DispatchMode::force_direct;
    if (config.mode == DispatchMode::automatic) {
        if (pairs <= config.fixed_direct_pair_limit ||
            std::min(a_span, b_span) <= 60) {
            use_direct = true;
        } else {
            const unsigned log_transform =
                static_cast<unsigned>(std::bit_width(transform_length) - 1U);
            const unsigned __int128 transform_budget =
                static_cast<unsigned __int128>(config.transform_cost_factor) *
                prime_count * transform_length * log_transform;
            use_direct = static_cast<unsigned __int128>(pairs) <= transform_budget;
        }
    }

    const bool singleton = a_end == 1 || b_end == 1;
    if (singleton && config.mode != DispatchMode::force_ntt) use_direct = true;
    const bool untruncated =
        static_cast<std::int64_t>(a[a_end - 1].score) +
                static_cast<std::int64_t>(b[b_end - 1].score) <
            limit;

    if (use_direct) {
        if (singleton)
            add_stat(accounting.singleton_calls, 1, "singleton convolution counter");
        else
            add_stat(accounting.direct_calls, 1, "direct convolution counter");
        add_stat(accounting.direct_pairs, pairs, "direct convolution pair counter");
        ScoreVec output = direct_convolution(
            a, a_end, b, b_end, limit, coefficient_bound);
        validate_output_mass(output, coefficient_bound, untruncated);
        return output;
    }

    if (transform_length > MAX_COMMON_TRANSFORM)
        throw std::length_error("score convolution exceeds the common 2^24 NTT limit");
    accounting.max_transform_length =
        std::max(accounting.max_transform_length, transform_length);
    add_stat(accounting.transform_points_times_primes,
             checked_mul(static_cast<Count>(transform_length), prime_count,
                         "transform work counter product"),
             "transform work counter accumulation");

    const auto residue1 = residue_convolution<PRIME1>(
        a, a_end, a_offset, a_span, b, b_end, b_offset, b_span);
    const auto residue2 = residue_convolution<PRIME2>(
        a, a_end, a_offset, a_span, b, b_end, b_offset, b_span);
    std::vector<std::uint32_t> residue3;
    if (prime_count == 2) {
        add_stat(accounting.ntt2_calls, 1, "two-prime NTT counter");
    } else {
        residue3 = residue_convolution<PRIME3>(
            a, a_end, a_offset, a_span, b, b_end, b_offset, b_span);
        add_stat(accounting.ntt3_calls, 1, "three-prime NTT counter");
    }
    if (residue1.size() != full_length || residue2.size() != full_length ||
        (prime_count == 3 && residue3.size() != full_length))
        throw std::logic_error("NTT convolution returned an unexpected length");

    const std::size_t retained = std::min(
        full_length,
        static_cast<std::size_t>(limit - (a_offset + b_offset)));
    add_stat(accounting.crt_coefficients, static_cast<Count>(retained),
             "CRT coefficient counter");
    ScoreVec output;
    for (std::size_t i = 0; i < retained; ++i) {
        const unsigned __int128 reconstructed =
            prime_count == 2
                ? static_cast<unsigned __int128>(
                      reconstruct_two(residue1[i], residue2[i]))
                : reconstruct_three(residue1[i], residue2[i], residue3[i]);
        if (reconstructed > coefficient_bound ||
            reconstructed > std::numeric_limits<Count>::max())
            throw std::logic_error("CRT coefficient exceeds mass certificate");
        const Count coefficient = static_cast<Count>(reconstructed);
        if (coefficient != 0)
            output.push_back(
                {a_offset + b_offset + static_cast<int>(i), coefficient});
    }
    validate_output_mass(output, coefficient_bound, untruncated);
    return output;
}

}
