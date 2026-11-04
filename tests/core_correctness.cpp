// Distribution and boundary checks with the covariance of overlapping adjacent-call pairs.
#include "core_test_helpers.hpp"

#include <array>
#include <iomanip>
#include <iostream>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

struct CoreSummary {
    bool heavy_ok = true;
    bool zero_mass_ok = true;
    bool rank_ok = true;
    bool weighted_ok = true;
    bool subset_ok = true;
    double heavy_max_weight_error = 0.0;
    double weighted_max_abs = 0.0;
    double weighted_max_z = 0.0;
    double marginal_max_abs = 0.0;
    double marginal_max_z = 0.0;
    double pair_max_abs = 0.0;
    double pair_max_z = 0.0;
    double lag1_max_abs = 0.0;
    double lag1_max_z = 0.0;
    double subset_max_abs = 0.0;
    double subset_tv = 0.0;
    std::uint64_t weighted_trials = 0;
    std::uint64_t subset_trials = 0;
};

bool check_heavy(const Instance& in, CoreSummary& summary) {
    WeightedJoinIndex index(in);
    auto oracle = oracle_join(in);
    std::vector<DrawnOutput> expected;
    for (const auto& output : oracle)
        if (output.p > 0.5) expected.push_back(output);
    std::vector<DrawnOutput> actual;
    HeavyStats stats;
    index.enumerate_heavy(
        [&](const Output& ids, double probability) {
            actual.push_back({ids, probability});
        },
        stats);
    std::sort(expected.begin(), expected.end(), output_less);
    std::sort(actual.begin(), actual.end(), output_less);
    if (expected.size() != actual.size()) return false;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (expected[i].ids != actual[i].ids) return false;
        summary.heavy_max_weight_error = std::max(
            summary.heavy_max_weight_error,
            std::abs(expected[i].p - actual[i].p));
        if (!(actual[i].p > 0.5)) return false;
    }
    return true;
}

CoreSummary run_original_core_suite(std::uint64_t seed) {
    CoreSummary summary;
    std::vector<Instance> cases;
    cases.push_back(manual_chain3());
    cases.push_back(generate_instance(
        "branch", 3, 2, 2, "uniform", 0.95, seed + 1));
    cases.push_back(generate_instance(
        "chain", 6, 2, 2, "uniform", 0.95, seed + 2));
    cases.push_back(generate_instance(
        "branch", 6, 2, 2, "skew", 1.0, seed + 3));
    for (const auto& in : cases)
        summary.heavy_ok = summary.heavy_ok && check_heavy(in, summary);

    {
        Instance in = zero_mass_case();
        WeightedJoinIndex index(in);
        std::mt19937_64 rng(seed + 11);
        QueryStats stats;
        auto sample = sample_subset(index, rng, stats, true);
        summary.zero_mass_ok =
            index.mu() == 0.0 && sample.empty() && stats.proposals == 0 &&
            stats.heavy.outputs == 0;
    }

    Instance in = manual_chain3();
    WeightedJoinIndex index(in);
    auto oracle = oracle_join(in);
    std::sort(oracle.begin(), oracle.end(), output_less);
    std::unordered_map<Output, std::size_t, VecHash> position;
    for (std::size_t i = 0; i < oracle.size(); ++i)
        position.emplace(oracle[i].ids, i);
    const std::size_t join_size = oracle.size();
    require(join_size <= 63, "core correctness universe too large");

    {
        RankIndex rank(in);
        std::vector<Output> actual;
        for (std::uint64_t i = 0; i < rank.join_size(); ++i)
            actual.push_back(rank.access(i));
        std::sort(actual.begin(), actual.end());
        std::vector<Output> expected;
        for (const auto& output : oracle) expected.push_back(output.ids);
        std::sort(expected.begin(), expected.end());
        summary.rank_ok = rank.join_size() == oracle.size() && actual == expected;
        std::vector<DrawnOutput> grouped;
        rank.enumerate_weighted(
            [&](const Output& ids, double probability) {
                grouped.push_back({ids, probability});
            });
        std::sort(grouped.begin(), grouped.end(), output_less);
        if (grouped.size() != oracle.size()) {
            summary.rank_ok = false;
        } else {
            for (std::size_t i = 0; i < oracle.size(); ++i)
                if (grouped[i].ids != oracle[i].ids ||
                    grouped[i].p != oracle[i].p)
                    summary.rank_ok = false;
        }
    }

    constexpr std::uint64_t weighted_trials = 300000;
    summary.weighted_trials = weighted_trials;
    std::vector<std::uint64_t> draw_count(join_size, 0);
    std::mt19937_64 draw_rng(seed + 21);
    for (std::uint64_t q = 0; q < weighted_trials; ++q) {
        const DrawnOutput draw = index.draw(draw_rng);
        auto it = position.find(draw.ids);
        if (it == position.end()) {
            summary.weighted_ok = false;
            break;
        }
        ++draw_count[it->second];
    }
    for (std::size_t i = 0; i < join_size; ++i) {
        const double expected = oracle[i].p / index.mu();
        const double observed =
            static_cast<double>(draw_count[i]) / weighted_trials;
        const double error = std::abs(observed - expected);
        summary.weighted_max_abs =
            std::max(summary.weighted_max_abs, error);
        if (expected > 0.0 && expected < 1.0) {
            const double standard_error = std::sqrt(
                expected * (1.0 - expected) / weighted_trials);
            summary.weighted_max_z = std::max(
                summary.weighted_max_z, error / standard_error);
        } else if (error != 0.0) {
            summary.weighted_ok = false;
        }
    }
    if (summary.weighted_max_z > 7.5) summary.weighted_ok = false;

    constexpr std::uint64_t subset_trials = 250000;
    summary.subset_trials = subset_trials;
    std::vector<std::uint64_t> marginal(join_size, 0);
    std::vector<std::uint64_t> pairs(join_size * join_size, 0);
    std::vector<std::uint64_t> lag_joint(join_size, 0);
    std::vector<std::uint64_t> mask_count(std::size_t{1} << join_size, 0);
    std::vector<int> previous(join_size, 0);
    std::mt19937_64 subset_rng(seed + 31);
    for (std::uint64_t q = 0; q < subset_trials; ++q) {
        QueryStats stats;
        const auto sample = sample_subset(index, subset_rng, stats, true);
        std::vector<int> bits(join_size, 0);
        std::uint64_t mask = 0;
        for (const auto& ids : sample) {
            auto it = position.find(ids);
            if (it == position.end()) {
                summary.subset_ok = false;
                continue;
            }
            bits[it->second] = 1;
            mask |= std::uint64_t{1} << it->second;
        }
        ++mask_count[static_cast<std::size_t>(mask)];
        for (std::size_t i = 0; i < join_size; ++i) {
            marginal[i] += bits[i];
            if (q != 0) lag_joint[i] += bits[i] && previous[i];
            for (std::size_t j = i + 1; j < join_size; ++j)
                pairs[i * join_size + j] += bits[i] && bits[j];
        }
        previous = std::move(bits);
    }
    for (std::size_t i = 0; i < join_size; ++i) {
        const double probability = oracle[i].p;
        const double observed =
            static_cast<double>(marginal[i]) / subset_trials;
        const double error = std::abs(observed - probability);
        summary.marginal_max_abs =
            std::max(summary.marginal_max_abs, error);
        if (probability > 0.0 && probability < 1.0) {
            const double standard_error = std::sqrt(
                probability * (1.0 - probability) / subset_trials);
            summary.marginal_max_z = std::max(
                summary.marginal_max_z, error / standard_error);
        } else if (error != 0.0) {
            summary.subset_ok = false;
        }
        const double expected_lag = probability * probability;
        const double observed_lag =
            static_cast<double>(lag_joint[i]) / (subset_trials - 1);
        const double lag_error = std::abs(observed_lag - expected_lag);
        summary.lag1_max_abs =
            std::max(summary.lag1_max_abs, lag_error);
        if (expected_lag > 0.0 && expected_lag < 1.0) {

            const double pair_count = static_cast<double>(subset_trials - 1);
            const double lag_sum_variance =
                pair_count * expected_lag * (1.0 - expected_lag) +
                2.0 * (pair_count - 1.0) * expected_lag * probability *
                    (1.0 - probability);
            const double standard_error =
                std::sqrt(lag_sum_variance) / pair_count;
            summary.lag1_max_z = std::max(
                summary.lag1_max_z, lag_error / standard_error);
        }
        for (std::size_t j = i + 1; j < join_size; ++j) {
            const double expected_pair = probability * oracle[j].p;
            const double observed_pair =
                static_cast<double>(pairs[i * join_size + j]) /
                subset_trials;
            const double pair_error =
                std::abs(observed_pair - expected_pair);
            summary.pair_max_abs =
                std::max(summary.pair_max_abs, pair_error);
            if (expected_pair > 0.0 && expected_pair < 1.0) {
                const double standard_error = std::sqrt(
                    expected_pair * (1.0 - expected_pair) /
                    subset_trials);
                summary.pair_max_z = std::max(
                    summary.pair_max_z, pair_error / standard_error);
            } else if (pair_error != 0.0) {
                summary.subset_ok = false;
            }
        }
    }
    for (std::size_t mask = 0; mask < mask_count.size(); ++mask) {
        double expected = 1.0;
        for (std::size_t i = 0; i < join_size; ++i)
            expected *= (mask & (std::size_t{1} << i))
                ? oracle[i].p : (1.0 - oracle[i].p);
        const double observed =
            static_cast<double>(mask_count[mask]) / subset_trials;
        const double error = std::abs(observed - expected);
        summary.subset_max_abs =
            std::max(summary.subset_max_abs, error);
        summary.subset_tv += error * 0.5;
    }
    if (summary.marginal_max_z > 7.5 || summary.pair_max_z > 7.5 ||
        summary.lag1_max_z > 7.5 || summary.subset_tv > 0.015 ||
        summary.subset_max_abs > 0.006)
        summary.subset_ok = false;
    return summary;
}

void verify_partition(const Instance& in) {
    WeightedJoinIndex index(in);
    const auto oracle = oracle_join(in);
    std::unordered_map<Output, double, VecHash> expected;
    for (const auto& output : oracle)
        if (output.p > 0.5) expected.emplace(output.ids, output.p);
    std::unordered_map<Output, double, VecHash> actual;
    HeavyStats stats;
    index.enumerate_heavy(
        [&](const Output& ids, double probability) {
            require(actual.emplace(ids, probability).second,
                    "duplicate heavy identity");
        },
        stats);
    require(actual == expected, "heavy partition/probability mismatch");
}

struct BoundarySummary {
    std::uint64_t structural_cases = 0;
    std::uint64_t single_included = 0;
    double marginal_max_z = 0.0;
    double pair_max_z = 0.0;
    double total_variation = 0.0;
};

BoundarySummary run_boundary_suite() {
    BoundarySummary summary;
    std::mt19937_64 rng(0x17a937ULL);
    for (const std::string shape : {"chain", "branch"}) {
        for (int m : {3, 6, 12, 32}) {
            for (int trial = 0; trial < 200; ++trial) {
                Instance in = generate_instance(
                    shape, m, 1, 1, "uniform", 1.0, 1, false);
                double prefix = 1.0;
                for (int i = 0; i < m - 1; ++i) {
                    const double weight =
                        1.0 - 0.4 * uniform_open(rng) / m;
                    in.rel[static_cast<std::size_t>(i)].tuples[0].weight =
                        weight;
                    prefix *= weight;
                }
                const double center = 0.5 / prefix;
                for (int ulps = -4; ulps <= 4; ++ulps) {
                    double weight = center;
                    for (int j = 0; j < std::abs(ulps); ++j)
                        weight = std::nextafter(
                            weight, ulps < 0 ? 0.0 : 1.0);
                    in.rel[static_cast<std::size_t>(m - 1)].tuples[0].weight =
                        weight;
                    verify_partition(in);
                    ++summary.structural_cases;
                }
            }
        }
    }
    for (const std::string shape : {"chain", "branch"}) {
        for (int m : {3, 6}) {
            for (int trial = 0; trial < 25; ++trial) {
                Instance in = generate_instance(
                    shape, m, 2, 2, "uniform", 1.0, 1, false);
                double prefix = 1.0;
                for (int i = 0; i < m - 1; ++i) {
                    const double weight =
                        1.0 - 0.4 * uniform_open(rng) / m;
                    for (auto& tuple :
                         in.rel[static_cast<std::size_t>(i)].tuples)
                        tuple.weight = weight;
                    prefix *= weight;
                }
                const double center = 0.5 / prefix;
                int offset = -4;
                for (auto& tuple :
                     in.rel[static_cast<std::size_t>(m - 1)].tuples) {
                    tuple.weight = center;
                    for (int j = 0; j < std::abs(offset); ++j)
                        tuple.weight = std::nextafter(
                            tuple.weight, offset < 0 ? 0.0 : 1.0);
                    offset = (offset + 1) % 5;
                }
                verify_partition(in);
                ++summary.structural_cases;
            }
        }
    }

    Instance in = generate_instance(
        "chain", 3, 1, 1, "uniform", 1.0, 1, false);
    in.rel[0].tuples[0].weight = 0.9169340963554261;
    in.rel[1].tuples[0].weight = 0.9992149193767528;
    in.rel[2].tuples[0].weight = 0.5457238974081999;
    verify_partition(in);
    ++summary.structural_cases;
    {
        WeightedJoinIndex index(in);
        std::mt19937_64 probe_rng(913);
        (void)index.draw(probe_rng);
        for (int q = 0; q < 100000; ++q) {
            QueryStats stats;
            summary.single_included +=
                !sample_subset(index, probe_rng, stats, true).empty();
        }
        require(std::abs(static_cast<double>(summary.single_included) -
                         50000.0) < 1200.0,
                "single-output boundary marginal failed");
    }

    const Tuple base = in.rel[2].tuples[0];
    in.rel[2].tuples.clear();
    for (int offset = -4; offset <= 4; ++offset) {
        Tuple tuple = base;
        tuple.id = 2 + offset + 4;
        for (int j = 0; j < std::abs(offset); ++j)
            tuple.weight = std::nextafter(
                tuple.weight, offset < 0 ? 0.0 : 1.0);
        in.rel[2].tuples.push_back(tuple);
    }
    verify_partition(in);
    ++summary.structural_cases;
    const auto oracle = oracle_join(in);
    const std::size_t n = oracle.size();
    require(n == 9, "unexpected boundary full-law size");
    std::unordered_map<Output, std::size_t, VecHash> position;
    for (std::size_t i = 0; i < n; ++i)
        position.emplace(oracle[i].ids, i);
    WeightedJoinIndex index(in);
    constexpr std::uint64_t trials = 150000;
    std::vector<std::uint64_t> marginal(n, 0), pairs(n * n, 0);
    std::vector<std::uint64_t> masks(std::size_t{1} << n, 0);
    for (std::uint64_t q = 0; q < trials; ++q) {
        QueryStats stats;
        const auto sample = sample_subset(index, rng, stats, true);
        std::uint64_t mask = 0;
        for (const auto& ids : sample) {
            auto it = position.find(ids);
            require(it != position.end(), "unexpected boundary identity");
            mask |= std::uint64_t{1} << it->second;
        }
        ++masks[static_cast<std::size_t>(mask)];
        for (std::size_t i = 0; i < n; ++i) {
            if (!(mask & (std::uint64_t{1} << i))) continue;
            ++marginal[i];
            for (std::size_t j = i + 1; j < n; ++j)
                if (mask & (std::uint64_t{1} << j)) ++pairs[i * n + j];
        }
    }
    auto z_score = [&](std::uint64_t observed, double probability) {
        return std::abs(static_cast<double>(observed) - trials * probability) /
               std::sqrt(trials * probability * (1.0 - probability));
    };
    for (std::size_t i = 0; i < n; ++i) {
        summary.marginal_max_z = std::max(
            summary.marginal_max_z, z_score(marginal[i], oracle[i].p));
        for (std::size_t j = i + 1; j < n; ++j)
            summary.pair_max_z = std::max(
                summary.pair_max_z,
                z_score(pairs[i * n + j], oracle[i].p * oracle[j].p));
    }
    for (std::size_t mask = 0; mask < masks.size(); ++mask) {
        double probability = 1.0;
        for (std::size_t i = 0; i < n; ++i)
            probability *= (mask & (std::size_t{1} << i))
                ? oracle[i].p : (1.0 - oracle[i].p);
        summary.total_variation += std::abs(
            static_cast<double>(masks[mask]) / trials - probability) / 2.0;
    }
    require(summary.marginal_max_z < 7.5 &&
            summary.pair_max_z < 7.5 && summary.total_variation < 0.07,
            "boundary full-law failed");
    return summary;
}

template<class Function>
void require_reject(Function&& function, const std::string& label) {
    bool rejected = false;
    try {
        function();
    } catch (const std::invalid_argument&) {
        rejected = true;
    } catch (const std::overflow_error&) {
        rejected = true;
    } catch (const std::length_error&) {
        rejected = true;
    }
    require(rejected, "expected explicit rejection: " + label);
}

Instance large_star(int relation_count, bool triple_first_child) {
    Instance in;
    in.m = relation_count;
    in.shape = "star";
    in.weight_mode = "mass_guard";
    in.parent.assign(static_cast<std::size_t>(relation_count), 0);
    in.parent[0] = -1;
    in.children.assign(static_cast<std::size_t>(relation_count), {});
    in.rel.resize(static_cast<std::size_t>(relation_count));
    in.preorder.reserve(static_cast<std::size_t>(relation_count));
    in.preorder.push_back(0);
    Tuple root;
    root.id = 0;
    root.weight = 1.0;
    root.edge_key.assign(static_cast<std::size_t>(relation_count), -1);
    for (int v = 1; v < relation_count; ++v) {
        in.parent[static_cast<std::size_t>(v)] = 0;
        in.children[0].push_back(v);
        in.preorder.push_back(v);
        root.edge_key[static_cast<std::size_t>(v)] = v;
        const int copies = triple_first_child && v == 1 ? 3 : 2;
        for (int j = 0; j < copies; ++j) {
            Tuple tuple;
            tuple.id = j;
            tuple.weight = 1.0;
            tuple.edge_key.assign(static_cast<std::size_t>(relation_count), -1);
            tuple.edge_key[0] = v;
            in.rel[static_cast<std::size_t>(v)].tuples.push_back(
                std::move(tuple));
        }
    }
    in.rel[0].tuples.push_back(std::move(root));
    return in;
}

std::uint64_t check_rejections() {
    std::uint64_t checks = 0;
    auto reject_instance = [&](Instance in, const std::string& label) {
        require_reject([&]() { WeightedJoinIndex index(in); }, label);
        ++checks;
    };
    {
        Instance in = manual_chain3();
        in.rel[0].tuples[0].weight =
            std::numeric_limits<double>::quiet_NaN();
        reject_instance(std::move(in), "NaN weight");
    }
    {
        Instance in = manual_chain3();
        in.rel[0].tuples[0].weight =
            std::numeric_limits<double>::infinity();
        reject_instance(std::move(in), "infinite weight");
    }
    {
        Instance in = manual_chain3();
        in.rel[0].tuples[1].id = in.rel[0].tuples[0].id;
        reject_instance(std::move(in), "duplicate tuple id");
    }
    {
        Instance in = manual_chain3();
        in.rel[1].tuples[0].edge_key.pop_back();
        reject_instance(std::move(in), "edge-key width");
    }
    {
        Instance in = manual_chain3();
        in.preorder = {0, 2, 1};
        reject_instance(std::move(in), "child-before-parent preorder");
    }
    {
        Instance in = manual_chain3();
        in.children[0].push_back(1);
        reject_instance(std::move(in), "duplicate child");
    }
    {
        Instance in = large_star(1025, true);
        require_reject([&]() { WeightedJoinIndex index(in); },
                       "nonfinite weighted mass");
        ++checks;
    }
    {
        Instance in = large_star(1024, true);
        WeightedJoinIndex index(in);
        require(std::isfinite(index.mu()) && index.mu() > 0.0,
                "finite huge mass fixture");
        std::mt19937_64 rng(7);
        QueryStats stats;
        require_reject([&]() { (void)sample_subset(index, rng, stats); },
                       "nonfinite Poisson rate");
        ++checks;
    }
    return checks;
}

}  

int main() {
    try {
        constexpr std::uint64_t seed = 0x5eed1234ULL;
        const CoreSummary core = run_original_core_suite(seed);
        require(core.heavy_ok && core.zero_mass_ok && core.rank_ok &&
                core.weighted_ok && core.subset_ok,
                "original core suite failed");
        const BoundarySummary boundary = run_boundary_suite();
        const std::uint64_t rejection_checks = check_rejections();

        std::cout << std::boolalpha << std::setprecision(17);
        std::cout << "CORE_CORRECTNESS PASS\n";
        std::cout << "original heavy_ok=" << core.heavy_ok
                  << " zero_mass_ok=" << core.zero_mass_ok
                  << " rank_ok=" << core.rank_ok
                  << " weighted_ok=" << core.weighted_ok
                  << " subset_ok=" << core.subset_ok
                  << " weighted_trials=" << core.weighted_trials
                  << " weighted_max_z=" << core.weighted_max_z
                  << " subset_trials=" << core.subset_trials
                  << " marginal_max_z=" << core.marginal_max_z
                  << " pair_max_z=" << core.pair_max_z
                  << " lag1_max_z=" << core.lag1_max_z
                  << " subset_tv=" << core.subset_tv << '\n';
        std::cout << "boundary structural_cases="
                  << boundary.structural_cases
                  << " single_trials=100000 single_included="
                  << boundary.single_included
                  << " full_law_trials=150000 marginal_max_z="
                  << boundary.marginal_max_z
                  << " pair_max_z=" << boundary.pair_max_z
                  << " subset_tv=" << boundary.total_variation << '\n';
        std::cout << "rejections checks=" << rejection_checks
                  << " input_validation=1 nonfinite_mass=1 nonfinite_rate=1\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "CORE_CORRECTNESS FAIL: " << error.what() << '\n';
        return 1;
    }
}
