// Validation helpers for the join-sampling implementation.
#include "orient.hpp"
#include <iostream>
#include <map>

using Law = std::map<Output, long double>;

static Law cartesian_law(const UndirectedInstance& input) {
    Law law;
    std::vector<const UndirectedTuple*> selected(input.m, nullptr);
    Output ids(input.m);
    auto key = [](const UndirectedTuple& row, EdgeId id) {
        for (const auto& entry : row.incident_keys)
            if (entry.first == id) return entry.second;
        throw std::runtime_error("oracle missing edge");
    };
    std::function<void(int, long double)> visit = [&](int v, long double p) {
        if (v == input.m) {
            for (const auto& e : input.edges)
                if (key(*selected[e.left_relation], e.id) !=
                    key(*selected[e.right_relation], e.id)) return;
            law.emplace(ids, p);
            return;
        }
        for (const auto& row : input.rel[v].tuples) {
            selected[v] = &row;
            ids[v] = row.id;
            visit(v + 1, p * row.weight);
        }
    };
    visit(0, 1);
    return law;
}

static UndirectedInstance fixture(bool branch, int m) {
    UndirectedInstance input;
    input.m = m;
    input.rel.resize(m);
    for (int v = 1; v < m; ++v)
        input.edges.push_back({7*v, branch ? (v-1)/2 : v-1, v});
    for (int v = 0; v < m; ++v) {
        for (int j = 0; j < 3; ++j) {
            UndirectedTuple row;
            row.id = 17 + j*101;
            row.weight = j == 0 ? 1.0 : (j == 1 ? .81 : .25);
            for (const auto& e : input.edges) {
                if (e.left_relation == v || e.right_relation == v)
                    row.incident_keys.emplace_back(e.id, j == 0 ? 0 : 100000000007LL);
            }
            std::reverse(row.incident_keys.begin(), row.incident_keys.end());
            input.rel[v].tuples.push_back(std::move(row));
        }
    }
    return input;
}

static void check(const UndirectedInstance& source) {
    const Law expected = cartesian_law(source);
    long double mass = 0, maximum = 0;
    for (const auto& item : expected) {
        mass += item.second;
        maximum = std::max(maximum, item.second);
    }
    for (int root = 0; root < source.m; ++root) {
        Instance directed = orient_instance(source, root);
        WeightedJoinIndex index(directed);
        if (std::abs(index.mu() - mass) > 1e-12L * std::max(1.L, mass))
            throw std::runtime_error("root changed mass");
        if (std::abs(index.max_probability() - maximum) > 1e-12L)
            throw std::runtime_error("root changed maximum");
        Law heavy;
        HeavyStats stats;
        index.enumerate_heavy([&](const Output& ids, double p) { heavy.emplace(ids,p); }, stats);
        std::size_t heavy_count = 0;
        for (const auto& item : expected) {
            if (item.second <= .5L) continue;
            ++heavy_count;
            auto found = heavy.find(item.first);
            if (found == heavy.end() || std::abs(found->second-item.second) > 1e-12L)
                throw std::runtime_error("root changed heavy identities");
        }
        if (heavy_count != heavy.size()) throw std::runtime_error("extra heavy output");
        std::mt19937_64 rng(47 + root);
        for (int trial = 0; trial < 100 && mass > 0; ++trial) {
            auto draw = index.draw(rng);
            const auto item = expected.find(draw.ids);
            if (item == expected.end() || std::abs(item->second-draw.p) > 1e-12L)
                throw std::runtime_error("invalid rerooted witness");
        }
    }
}

static void rejects(const UndirectedInstance& source, int root) {
    try { auto x = orient_instance(source, root); (void)x; }
    catch (const std::exception&) { return; }
    throw std::runtime_error("malformed undirected input accepted");
}

int main() {
    try {
        for (int m : {1, 3, 4, 6}) for (bool branch : {false,true}) check(fixture(branch,m));
        auto empty = fixture(false,4); empty.rel[2].tuples.clear(); check(empty);
        auto dangling = fixture(false,4);
        dangling.rel[1].tuples[1].incident_keys[0].second = -500;
        check(dangling);
        auto bad = fixture(false,4); rejects(bad,-1); rejects(bad,4);
        bad.edges[1].id = bad.edges[0].id; rejects(bad,0);
        bad = fixture(false,4); bad.edges[2] = {21,0,2}; rejects(bad,0);
        bad = fixture(false,4); bad.rel[1].tuples[0].incident_keys.pop_back(); rejects(bad,0);
        bad = fixture(false,4); bad.rel[1].tuples[0].incident_keys[1] = bad.rel[1].tuples[0].incident_keys[0]; rejects(bad,0);
        bad = fixture(false,4); bad.rel[0].tuples[0].incident_keys[0].first=21; rejects(bad,0);
        bad = fixture(false,4); bad.rel[0].tuples[1].id=bad.rel[0].tuples[0].id; rejects(bad,0);
        bad = fixture(false,4); bad.rel[0].tuples[0].weight=std::numeric_limits<double>::infinity(); rejects(bad,0);
        std::cout << "PASS all-root mass/max/heavy/witness invariance, empty/dangling, malformed inputs\n";
    } catch (const std::exception& e) {
        std::cerr << "FAIL " << e.what() << '\n'; return 1;
    }
}
