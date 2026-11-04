// Validation helpers for the join-sampling implementation.
#include "core_test_helpers.hpp"
#include "rank_scan_controls.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_map>

namespace {

using rank_scan_controls::Count;
using rank_scan_controls::MaterializedScanSampler;
using rank_scan_controls::ScanQueryStats;
using rank_scan_controls::UniformQueryStats;
using rank_scan_controls::UniformRankSampler;

void require(bool condition, const std::string& message) {
    if (!condition)
        throw std::runtime_error("rank/scan control test failure: " + message);
}

template<class Exception, class Function>
void require_exception(Function&& function, const std::string& message) {
    bool threw = false;
    try {
        function();
    } catch (const Exception&) {
        threw = true;
    }
    require(threw, message);
}

std::vector<Output> sorted_oracle_ids(const Instance& instance) {
    std::vector<Output> ids;
    for (const auto& row : oracle_join(instance)) ids.push_back(row.ids);
    std::sort(ids.begin(), ids.end());
    return ids;
}

void set_relation_weight(Instance& instance, int relation, double weight) {
    for (auto& tuple : instance.rel[static_cast<std::size_t>(relation)].tuples)
        tuple.weight = weight;
}

Instance uniform_case(double root_weight, double middle_weight,
                      double leaf_weight) {
    Instance instance = generate_instance(
        "chain", 3, 2, 2, "uniform", 1.0, 501, true);
    set_relation_weight(instance, 0, root_weight);
    set_relation_weight(instance, 1, middle_weight);
    set_relation_weight(instance, 2, leaf_weight);
    instance.weight_mode = "relation_constant";
    return instance;
}

Instance shared_dangling_case() {
    Instance instance;
    instance.m = 3;
    instance.shape = "chain";
    instance.weight_mode = "uniform";
    instance.parent = {-1, 0, 1};
    instance.children = {{1}, {2}, {}};
    instance.preorder = {0, 1, 2};
    instance.rel.resize(3);

    for (int i = 0; i < 64; ++i) {
        Tuple tuple;
        tuple.id = i;
        tuple.weight = 0.5;
        tuple.edge_key.assign(3, -1);
        tuple.edge_key[1] = 10;
        instance.rel[0].tuples.push_back(std::move(tuple));
    }
    for (int i = 0; i < 256; ++i) {
        Tuple tuple;
        tuple.id = 1000 + i;
        tuple.weight = 0.5;
        tuple.edge_key.assign(3, -1);
        tuple.edge_key[0] = 10;
        tuple.edge_key[2] = 10000 + i;
        instance.rel[1].tuples.push_back(std::move(tuple));
    }
    Tuple middle;
    middle.id = 2000;
    middle.weight = 0.5;
    middle.edge_key.assign(3, -1);
    middle.edge_key[0] = 10;
    middle.edge_key[2] = 20;
    instance.rel[1].tuples.push_back(std::move(middle));

    Tuple leaf;
    leaf.id = 3000;
    leaf.weight = 0.5;
    leaf.edge_key.assign(3, -1);
    leaf.edge_key[1] = 20;
    instance.rel[2].tuples.push_back(std::move(leaf));
    return instance;
}

Instance wide_star(int children, int copies) {
    Instance instance;
    instance.m = children + 1;
    instance.shape = "branch";
    instance.weight_mode = "uniform";
    instance.parent.assign(static_cast<std::size_t>(instance.m), 0);
    instance.parent[0] = -1;
    instance.children.assign(static_cast<std::size_t>(instance.m), {});
    instance.rel.resize(static_cast<std::size_t>(instance.m));
    Tuple root;
    root.id = 0;
    root.weight = 0.5;
    root.edge_key.assign(static_cast<std::size_t>(instance.m), -1);
    int next_id = 1;
    for (int child = 1; child < instance.m; ++child) {
        instance.children[0].push_back(child);
        const Key key = 1000 + child;
        root.edge_key[static_cast<std::size_t>(child)] = key;
        for (int copy = 0; copy < copies; ++copy) {
            Tuple tuple;
            tuple.id = next_id++;
            tuple.weight = 0.5;
            tuple.edge_key.assign(static_cast<std::size_t>(instance.m), -1);
            tuple.edge_key[0] = key;
            instance.rel[static_cast<std::size_t>(child)].tuples.push_back(
                std::move(tuple));
        }
    }
    instance.rel[0].tuples.push_back(std::move(root));
    instance.preorder = make_preorder(instance.children);
    return instance;
}

Instance materialization_limit_case() {
    Instance instance;
    instance.m = 3;
    instance.shape = "branch";
    instance.weight_mode = "uniform";
    instance.parent = {-1, 0, 0};
    instance.children = {{1, 2}, {}, {}};
    instance.preorder = {0, 1, 2};
    instance.rel.resize(3);
    Tuple root;
    root.id = 0;
    root.weight = 0.5;
    root.edge_key = {-1, 11, 22};
    instance.rel[0].tuples.push_back(std::move(root));
    for (int relation = 1; relation <= 2; ++relation) {
        for (int i = 0; i < 1001; ++i) {
            Tuple tuple;
            tuple.id = relation * 2000 + i;
            tuple.weight = 0.5;
            tuple.edge_key.assign(3, -1);
            tuple.edge_key[0] = relation == 1 ? 11 : 22;
            instance.rel[static_cast<std::size_t>(relation)].tuples.push_back(
                std::move(tuple));
        }
    }
    return instance;
}

void check_uniform_identity_and_edges() {
    Instance instance = uniform_case(0.8, 0.6, 0.5);
    UniformRankSampler sampler(instance);
    const auto want = sorted_oracle_ids(instance);
    require(sampler.join_size() == want.size(), "uniform join size");
    require(sampler.output_probability() == 0.8 * 0.6 * 0.5,
            "uniform relation-product probability");
    std::vector<Output> got;
    for (Count rank = 0; rank < sampler.join_size(); ++rank)
        got.push_back(sampler.access_rank(rank));
    std::sort(got.begin(), got.end());
    require(got == want, "uniform rank bijection with dangling rows");
    require(sampler.approximate_bytes() > 0, "uniform byte accounting");

    {
        Instance all_one = uniform_case(1.0, 1.0, 1.0);
        UniformRankSampler deterministic(all_one);
        std::mt19937_64 rng(1);
        UniformQueryStats stats;
        auto sample = deterministic.sample_vector(rng, stats);
        require(sample.size() == deterministic.join_size(), "uniform p=1 size");
        require(stats.selected_ranks == deterministic.join_size(),
                "uniform p=1 rank stats");
        for (Count rank = 0; rank < deterministic.join_size(); ++rank)
            require(sample[static_cast<std::size_t>(rank)] ==
                        deterministic.access_rank(rank),
                    "uniform p=1 rank order");
    }
    {
        Instance all_zero = uniform_case(1.0, 0.0, 1.0);
        UniformRankSampler deterministic(all_zero);
        std::mt19937_64 rng(2);
        UniformQueryStats stats;
        require(deterministic.sample_vector(rng, stats).empty(),
                "uniform p=0 empty sample");
        require(stats.selected_ranks == 0, "uniform p=0 rank stats");
    }
    {
        Instance nonuniform = instance;
        nonuniform.rel[1].tuples.front().weight = 0.61;
        require_exception<std::invalid_argument>([&] {
            UniformRankSampler rejected(nonuniform);
            (void)rejected;
        }, "nonuniform relation weights rejected");
    }
    require_exception<rank_scan_controls::UnsupportedUniformPopulation>([&] {
        Instance too_large = wide_star(54, 2);
        UniformRankSampler rejected(too_large);
        (void)rejected;
    }, "uniform rank population above 2^53 rejected");
}

void check_scan_identity_and_edges() {
    Instance instance = manual_chain3();
    MaterializedScanSampler sampler(instance);
    auto want_rows = oracle_join(instance);
    require(sampler.join_size() == want_rows.size(), "scan join size");
    std::vector<Output> got;
    double expected = 0.0;
    for (const auto& row : sampler.rows()) {
        got.push_back(row.ids);
        expected += row.probability;
    }
    std::sort(got.begin(), got.end());
    require(got == sorted_oracle_ids(instance), "scan materialized identity");
    require(std::abs(expected - sampler.expected_size()) <= 1e-15,
            "scan expected size");
    require(sampler.approximate_bytes() >= sizeof(MaterializedScanSampler),
            "scan owned byte accounting");

    {
        Instance all_one = uniform_case(1.0, 1.0, 1.0);
        MaterializedScanSampler deterministic(all_one);
        std::mt19937_64 rng(3);
        ScanQueryStats stats;
        const auto sample = deterministic.sample_vector(rng, stats);
        require(sample.size() == deterministic.join_size(), "scan p=1 size");
        require(stats.rows_scanned == deterministic.join_size(),
                "scan p=1 scanned rows");
        require(stats.outputs == deterministic.join_size(),
                "scan p=1 output stats");
    }
    {
        Instance all_zero = uniform_case(0.0, 1.0, 1.0);
        MaterializedScanSampler deterministic(all_zero);
        std::mt19937_64 rng(4);
        ScanQueryStats stats;
        require(deterministic.sample_vector(rng, stats).empty(),
                "scan p=0 empty sample");
        require(stats.rows_scanned == deterministic.join_size(),
                "scan p=0 still scans all rows");
        require(stats.outputs == 0, "scan p=0 output stats");
    }
}

struct LawResult {
    double max_marginal_z = 0.0;
    double max_pair_z = 0.0;
};

double z_score(double observed, double expected, Count trials) {
    if (expected == 0.0 || expected == 1.0)
        return observed == expected ? 0.0
                                    : std::numeric_limits<double>::infinity();
    const double variance = expected * (1.0 - expected) /
                            static_cast<double>(trials);
    return std::abs(observed - expected) / std::sqrt(variance);
}

template<class Query>
LawResult check_law(const std::string& label,
                    const std::vector<DrawnOutput>& oracle,
                    Query&& query, Count trials) {
    std::unordered_map<Output, std::size_t, VecHash> position;
    for (std::size_t i = 0; i < oracle.size(); ++i)
        position.emplace(oracle[i].ids, i);
    std::vector<Count> marginal(oracle.size(), 0);
    std::vector<std::vector<Count>> pairs(
        oracle.size(), std::vector<Count>(oracle.size(), 0));
    for (Count trial = 0; trial < trials; ++trial) {
        const auto sample = query();
        std::vector<std::size_t> present;
        for (const auto& ids : sample) {
            auto it = position.find(ids);
            require(it != position.end(), label + ": unknown output");
            present.push_back(it->second);
        }
        std::sort(present.begin(), present.end());
        require(std::adjacent_find(present.begin(), present.end()) == present.end(),
                label + ": duplicate output");
        for (std::size_t i : present) ++marginal[i];
        for (std::size_t a = 0; a < present.size(); ++a)
            for (std::size_t b = a + 1; b < present.size(); ++b)
                ++pairs[present[a]][present[b]];
    }
    LawResult result;
    for (std::size_t i = 0; i < oracle.size(); ++i) {
        const double observed = static_cast<double>(marginal[i]) / trials;
        result.max_marginal_z = std::max(
            result.max_marginal_z, z_score(observed, oracle[i].p, trials));
        for (std::size_t j = i + 1; j < oracle.size(); ++j) {
            const double pair_observed = static_cast<double>(pairs[i][j]) / trials;
            result.max_pair_z = std::max(
                result.max_pair_z,
                z_score(pair_observed, oracle[i].p * oracle[j].p, trials));
        }
    }
    require(result.max_marginal_z < 8.0, label + ": marginal law");
    require(result.max_pair_z < 8.0, label + ": pair law");
    return result;
}

std::pair<LawResult, LawResult> check_statistical_laws() {
    constexpr Count trials = 200000;
    Instance uniform = uniform_case(0.8, 0.6, 0.5);
    UniformRankSampler uniform_sampler(uniform);
    auto uniform_oracle = oracle_join(uniform);
    std::mt19937_64 uniform_rng(0x0123456789abcdefULL);
    const LawResult uniform_law = check_law(
        "uniform geometric", uniform_oracle,
        [&] { return uniform_sampler.sample_vector(uniform_rng); }, trials);

    Instance arbitrary = manual_chain3();
    MaterializedScanSampler scan_sampler(arbitrary);
    auto scan_oracle = oracle_join(arbitrary);
    std::mt19937_64 scan_rng(0xfedcba9876543210ULL);
    const LawResult scan_law = check_law(
        "materialized scan", scan_oracle,
        [&] { return scan_sampler.sample_vector(scan_rng); }, trials);
    return {uniform_law, scan_law};
}

void check_compaction_and_limits() {
    {
        Instance dangling = shared_dangling_case();
        MaterializedScanSampler sampler(dangling);
        const auto& stats = sampler.build_stats();
        require(sampler.join_size() == 64, "shared dangling join size");
        require(stats.rank.discarded_zero_tuple_entries == 256,
                "shared dangling tuples compacted");
        require(stats.rank.positive_tuple_entries == 66,
                "shared dangling positive tuple entries");
        require(stats.enumeration.tuple_extensions == 3 * sampler.join_size(),
                "shared dangling enumeration charges exactly mJ");
    }
    require_exception<rank_scan_controls::UnsupportedMaterialization>([&] {
        Instance too_large = materialization_limit_case();
        MaterializedScanSampler rejected(too_large);
        (void)rejected;
    }, "materialization above one million reports unsupported");

    require_exception<std::overflow_error>([&] {
        Instance overflow = wide_star(64, 2);
        UniformRankSampler rejected(overflow);
        (void)rejected;
    }, "rank count above uint64 throws");
    require_exception<std::overflow_error>([&] {
        Instance overflow = wide_star(64, 2);
        MaterializedScanSampler rejected(overflow);
        (void)rejected;
    }, "scan count above uint64 throws");
}

}  

int main() {
    try {
        check_uniform_identity_and_edges();
        check_scan_identity_and_edges();
        const auto laws = check_statistical_laws();
        check_compaction_and_limits();
        std::cout << "RANK_SCAN_CONTROLS_VALIDATION PASS\n";
        std::cout << "uniform_law_trials=200000 max_marginal_z="
                  << laws.first.max_marginal_z
                  << " max_pair_z=" << laws.first.max_pair_z << '\n';
        std::cout << "scan_law_trials=200000 max_marginal_z="
                  << laws.second.max_marginal_z
                  << " max_pair_z=" << laws.second.max_pair_z << '\n';
        std::cout << "identity=2 p0_p1=4 nonuniform_reject=1 "
                     "population_reject=1 compaction=1 materialization_limit=1 "
                     "uint64_overflow=2\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "RANK_SCAN_CONTROLS_VALIDATION FAIL: "
                  << error.what() << '\n';
        return 1;
    }
}
