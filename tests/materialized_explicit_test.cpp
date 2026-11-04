// Validation helpers for the join-sampling implementation.
#include "core_test_helpers.hpp"
#include "materialized_explicit.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <vector>

namespace {

using materialized_explicit::BuildStats;
using materialized_explicit::Count;
using materialized_explicit::MaterializedExplicitSampler;
using materialized_explicit::QueryStats;

void require(bool condition, const std::string& message) {
    if (!condition)
        throw std::runtime_error(
            "materialized explicit test failure: " + message);
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

bool close(long double a, long double b, long double tolerance = 1e-15L) {
    return std::fabs(a - b) <=
        tolerance * std::max(1.0L, std::max(std::fabs(a), std::fabs(b)));
}

Instance one_relation(const std::vector<double>& probabilities) {
    Instance instance;
    instance.m = 1;
    instance.shape = "single";
    instance.weight_mode = "explicit";
    instance.parent = {-1};
    instance.children = {{}};
    instance.preorder = {0};
    instance.rel.resize(1);
    for (std::size_t i = 0; i < probabilities.size(); ++i) {
        Tuple tuple;
        tuple.id = 100 + static_cast<int>(i);
        tuple.weight = probabilities[i];
        tuple.edge_key = {-1};
        instance.rel[0].tuples.push_back(std::move(tuple));
    }
    return instance;
}

Instance two_relation_law_witness() {
    Instance instance;
    instance.m = 2;
    instance.shape = "chain";
    instance.weight_mode = "dyadic_cartesian";
    instance.parent = {-1, 0};
    instance.children = {{1}, {}};
    instance.preorder = {0, 1};
    instance.rel.resize(2);

    for (int i = 0; i < 2; ++i) {
        Tuple root;
        root.id = 100 + i;
        root.weight = i == 0 ? 0.25 : 0.75;
        root.edge_key = {-1, 700};
        instance.rel[0].tuples.push_back(std::move(root));

        Tuple child;
        child.id = 200 + i;
        child.weight = i == 0 ? 0.5 : 1.0;
        child.edge_key = {700, -1};
        instance.rel[1].tuples.push_back(std::move(child));
    }
    return instance;
}

Instance shared_dangling_case() {
    Instance instance;
    instance.m = 3;
    instance.shape = "chain";
    instance.weight_mode = "dangling";
    instance.parent = {-1, 0, 1};
    instance.children = {{1}, {2}, {}};
    instance.preorder = {0, 1, 2};
    instance.rel.resize(3);

    for (int i = 0; i < 4; ++i) {
        Tuple root;
        root.id = 10 + i;
        root.weight = 0.5;
        root.edge_key = {-1, 700, -1};
        instance.rel[0].tuples.push_back(std::move(root));
    }
    for (int i = 0; i < 10; ++i) {
        Tuple dangling;
        dangling.id = 100 + i;
        dangling.weight = 0.4;
        dangling.edge_key = {700, -1, 800 + i};
        instance.rel[1].tuples.push_back(std::move(dangling));
    }
    Tuple live;
    live.id = 200;
    live.weight = 0.4;
    live.edge_key = {700, -1, 900};
    instance.rel[1].tuples.push_back(std::move(live));

    Tuple leaf;
    leaf.id = 300;
    leaf.weight = 0.25;
    leaf.edge_key = {-1, 900, -1};
    instance.rel[2].tuples.push_back(std::move(leaf));
    return instance;
}

Instance above_budget_case() {
    constexpr int children = 25;
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
    root.weight = 1.0;
    root.edge_key.assign(static_cast<std::size_t>(instance.m), -1);
    for (int child = 1; child < instance.m; ++child) {
        instance.children[0].push_back(child);
        const Key key = 1000 + child;
        root.edge_key[static_cast<std::size_t>(child)] = key;
        for (int copy = 0; copy < 2; ++copy) {
            Tuple tuple;
            tuple.id = 10 * child + copy;
            tuple.weight = 1.0;
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

void verify_byte_accounting(const MaterializedExplicitSampler& sampler) {
    const BuildStats& stats = sampler.build_stats();
    require(stats.flat_id_cells ==
                static_cast<std::size_t>(sampler.join_size()) *
                    sampler.relation_count(),
            "flat ID cell count");
    require(stats.flat_id_bytes ==
                stats.flat_id_capacity * sizeof(int),
            "flat ID capacity bytes");
    require(stats.temporary_probability_cells ==
                static_cast<std::size_t>(sampler.join_size()),
            "transient probability cell count");
    require(stats.temporary_probability_bytes ==
                stats.temporary_probability_capacity * sizeof(double),
            "transient probability capacity bytes");
    require(stats.subset_index_bytes >=
                sizeof(poisson_join::ExplicitSubsetIndex),
            "nested subset index bytes");
    require(stats.subset_dynamic_bytes ==
                stats.subset_index_bytes -
                    sizeof(poisson_join::ExplicitSubsetIndex),
            "nested subset dynamic bytes");
    require(stats.owned_persistent_bytes ==
                sizeof(MaterializedExplicitSampler) + stats.flat_id_bytes +
                    stats.subset_dynamic_bytes,
            "persistent bytes count nested object once");
    require(sampler.approximate_bytes() == stats.owned_persistent_bytes,
            "persistent byte accessor");
}

void check_structure_and_moments() {
    const Instance instance = manual_chain3();
    const std::vector<DrawnOutput> oracle = oracle_join(instance);
    MaterializedExplicitSampler sampler(instance);

    require(sampler.join_size() == oracle.size(), "manual join size");
    require(sampler.relation_count() == 3, "manual relation count");
    require(sampler.build_stats().enumeration.outputs == sampler.join_size(),
            "enumeration output count");
    require(sampler.build_stats().enumeration.tuple_extensions <=
                3 * sampler.join_size(),
            "grouped enumeration mJ bound");

    std::vector<Output> want;
    std::vector<Output> got;
    long double mu = 0.0L;
    long double sum_p2 = 0.0L;
    double max_probability = 0.0;
    for (const DrawnOutput& row : oracle) {
        want.push_back(row.ids);
        mu += static_cast<long double>(row.p);
        sum_p2 += static_cast<long double>(row.p) * row.p;
        max_probability = std::max(max_probability, row.p);
    }
    for (Count i = 0; i < sampler.join_size(); ++i)
        got.push_back(sampler.access_output(i));
    std::sort(want.begin(), want.end());
    std::sort(got.begin(), got.end());
    require(got == want, "flat outputs equal explicit oracle");
    require(close(sampler.mu(), mu), "mu identity");
    require(close(sampler.sum_p2(), sum_p2), "sum p squared identity");
    require(sampler.max_probability() == max_probability,
            "maximum probability identity");
    require_exception<std::out_of_range>([&] {
        (void)sampler.access_output(sampler.join_size());
    }, "flat output bounds");
    verify_byte_accounting(sampler);
}

void check_edges_and_dangling() {
    {
        MaterializedExplicitSampler sampler(one_relation({0.0, 1.0}));
        std::mt19937_64 rng(1);
        QueryStats stats;
        const std::vector<Output> sample = sampler.sample_vector(rng, stats);
        require(sample == std::vector<Output>{{101}}, "p0/p1 identities");
        require(stats.selected_indices == 1 && stats.outputs == 1,
                "p0/p1 query counts");
        require(stats.selected_index_temporary_bytes >= sizeof(std::size_t),
                "selected-index temporary allocation");
        require(sampler.mu() == 1.0L && sampler.sum_p2() == 1.0L &&
                    sampler.max_probability() == 1.0,
                "p0/p1 moments");
        verify_byte_accounting(sampler);
    }
    {
        MaterializedExplicitSampler sampler(one_relation({}));
        std::mt19937_64 rng(2);
        QueryStats stats;
        require(sampler.join_size() == 0 && sampler.sample_vector(rng, stats).empty(),
                "empty join query");
        require(sampler.mu() == 0.0L && sampler.sum_p2() == 0.0L &&
                    sampler.max_probability() == 0.0,
                "empty moments");
        require(stats.outputs == 0 && stats.selected_indices == 0,
                "empty query counts");
        verify_byte_accounting(sampler);
    }
    {
        MaterializedExplicitSampler sampler(shared_dangling_case());
        const BuildStats& stats = sampler.build_stats();
        require(sampler.join_size() == 4, "dangling join size");
        require(stats.rank.discarded_zero_tuple_entries == 10,
                "dangling rows compacted");
        require(stats.enumeration.tuple_extensions == 3 * sampler.join_size(),
                "dangling grouped enumeration exact mJ charge");
        for (Count i = 0; i < sampler.join_size(); ++i) {
            const Output output = sampler.access_output(i);
            require(output.size() == 3 && output[1] == 200 && output[2] == 300,
                    "dangling output identity");
        }
    }
    require_exception<materialized_explicit::UnsupportedMaterializedExplicit>([&] {
        MaterializedExplicitSampler rejected(above_budget_case());
        (void)rejected;
    }, "population above predeclared study budget rejected");
}

double z_score(std::uint64_t count, double probability,
               std::uint64_t trials) {
    const double observed = static_cast<double>(count) / trials;
    const double variance = probability * (1.0 - probability) / trials;
    if (!(variance > 0.0)) return observed == probability ? 0.0 :
        std::numeric_limits<double>::infinity();
    return std::fabs(observed - probability) / std::sqrt(variance);
}

struct LawResult {
    double max_marginal_z = 0.0;
    double max_pair_z = 0.0;
    double max_full_z = 0.0;
};

LawResult check_subset_law() {
    const Instance instance = two_relation_law_witness();
    const std::vector<DrawnOutput> oracle = oracle_join(instance);
    require(oracle.size() == 4, "law Cartesian witness size");
    const std::vector<double> expected_probability = {0.125, 0.25, 0.375, 0.75};
    std::vector<double> probability;
    std::map<Output, std::size_t> bit_by_output;
    for (std::size_t i = 0; i < oracle.size(); ++i) {
        require(oracle[i].p == expected_probability[i],
                "law dyadic PRODUCT probability");
        require(bit_by_output.emplace(oracle[i].ids, i).second,
                "law oracle output uniqueness");
        probability.push_back(oracle[i].p);
    }
    MaterializedExplicitSampler sampler(instance);
    constexpr std::uint64_t trials = 200000;
    std::vector<std::uint64_t> marginal(4, 0);
    std::vector<std::vector<std::uint64_t>> pair(
        4, std::vector<std::uint64_t>(4, 0));
    std::vector<std::uint64_t> full(16, 0);
    std::mt19937_64 rng(0x6d61746578706cULL);

    for (std::uint64_t trial = 0; trial < trials; ++trial) {
        QueryStats stats;
        const std::vector<Output> sample = sampler.sample_vector(rng, stats);
        unsigned mask = 0;
        for (const Output& output : sample) {
            require(output.size() == 2, "law output arity");
            const auto found = bit_by_output.find(output);
            require(found != bit_by_output.end(), "law full output ID vector");
            const std::size_t bit = found->second;
            require((mask & (1U << bit)) == 0, "law duplicate output");
            mask |= 1U << bit;
        }
        require(stats.selected_indices == sample.size() &&
                    stats.outputs == sample.size(),
                "law query accounting");
        ++full[mask];
        for (std::size_t i = 0; i < 4; ++i) {
            if ((mask & (1U << i)) == 0) continue;
            ++marginal[i];
            for (std::size_t j = i + 1; j < 4; ++j)
                if (mask & (1U << j)) ++pair[i][j];
        }
    }

    LawResult result;
    for (std::size_t i = 0; i < 4; ++i) {
        result.max_marginal_z = std::max(
            result.max_marginal_z,
            z_score(marginal[i], probability[i], trials));
        for (std::size_t j = i + 1; j < 4; ++j)
            result.max_pair_z = std::max(
                result.max_pair_z,
                z_score(pair[i][j], probability[i] * probability[j], trials));
    }
    for (unsigned mask = 0; mask < 16; ++mask) {
        double cell_probability = 1.0;
        for (std::size_t i = 0; i < 4; ++i)
            cell_probability *= (mask & (1U << i))
                ? probability[i]
                : 1.0 - probability[i];
        result.max_full_z = std::max(
            result.max_full_z,
            z_score(full[mask], cell_probability, trials));
    }
    require(result.max_marginal_z < 8.0, "subset marginal law");
    require(result.max_pair_z < 8.0, "subset pair law");
    require(result.max_full_z < 8.0, "subset full law");
    return result;
}

}  

int main() {
    try {
        check_structure_and_moments();
        check_edges_and_dangling();
        const LawResult law = check_subset_law();
        std::cout << "MATERIALIZED_EXPLICIT_VALIDATION PASS\n";
        std::cout << "law_trials=200000 max_marginal_z="
                  << law.max_marginal_z
                  << " max_pair_z=" << law.max_pair_z
                  << " max_full_z=" << law.max_full_z << '\n';
        std::cout << "structure=1 p0_p1=1 empty=1 dangling=1 "
                     "budget_rejection=1 byte_accounting=1\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "MATERIALIZED_EXPLICIT_VALIDATION FAIL: "
                  << error.what() << '\n';
        return 1;
    }
}
