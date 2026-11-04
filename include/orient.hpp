// Root the input join tree while preserving tuple identities and edge keys.
#pragma once

#include "core.hpp"
#include <functional>

inline Instance orient_instance(const UndirectedInstance& source, int root) {
    if (source.m <= 0 || source.rel.size() != static_cast<std::size_t>(source.m))
        throw std::invalid_argument("undirected relation dimensions mismatch");
    if (root < 0 || root >= source.m)
        throw std::invalid_argument("root out of range");
    const std::size_t m = static_cast<std::size_t>(source.m);
    if (source.edges.size() != m - 1)
        throw std::invalid_argument("join tree must have m-1 edges");
    std::unordered_map<EdgeId, std::pair<int, int>> endpoints;
    std::vector<std::vector<int>> adjacent(m);
    for (const auto& e : source.edges) {
        const int u = e.left_relation, v = e.right_relation;
        if (e.id < 0 || u < 0 || v < 0 || u >= source.m || v >= source.m || u == v)
            throw std::invalid_argument("invalid undirected join edge");
        if (!endpoints.emplace(e.id, std::make_pair(u, v)).second)
            throw std::invalid_argument("duplicate join edge id");
        adjacent[u].push_back(v);
        adjacent[v].push_back(u);
    }
    for (auto& neighbors : adjacent) {
        std::sort(neighbors.begin(), neighbors.end());
        if (std::adjacent_find(neighbors.begin(), neighbors.end()) != neighbors.end())
            throw std::invalid_argument("parallel join tree edges");
    }
    Instance result;
    result.m = source.m;
    result.shape = source.shape;
    result.weight_mode = source.weight_mode;
    result.parent.assign(m, -2);
    result.children.resize(m);
    result.rel.resize(m);
    std::function<void(int, int)> visit = [&](int v, int parent) {
        if (result.parent[v] != -2)
            throw std::invalid_argument("cycle in join tree");
        result.parent[v] = parent;
        result.preorder.push_back(v);
        for (int u : adjacent[v]) {
            if (u == parent) continue;
            result.children[v].push_back(u);
            visit(u, v);
        }
    };
    visit(root, -1);
    if (result.preorder.size() != m)
        throw std::invalid_argument("disconnected join tree");
    for (int v = 0; v < source.m; ++v) {
        auto& rows = result.rel[v].tuples;
        rows.reserve(source.rel[v].tuples.size());
        for (const auto& input : source.rel[v].tuples) {
            if (input.incident_keys.size() != adjacent[v].size())
                throw std::invalid_argument("row incident-key degree mismatch");
            Tuple tuple;
            tuple.id = input.id;
            tuple.weight = input.weight;
            tuple.edge_key.assign(m, 0);
            std::vector<unsigned char> seen(m, 0);
            for (const auto& entry : input.incident_keys) {
                const auto it = endpoints.find(entry.first);
                if (it == endpoints.end())
                    throw std::invalid_argument("unknown row edge id");
                const auto uv = it->second;
                int neighbor;
                if (uv.first == v) neighbor = uv.second;
                else if (uv.second == v) neighbor = uv.first;
                else throw std::invalid_argument("row uses nonincident edge");
                if (seen[neighbor])
                    throw std::invalid_argument("duplicate row incident edge");
                seen[neighbor] = 1;
                tuple.edge_key[neighbor] = entry.second;
            }
            rows.push_back(std::move(tuple));
        }
    }
    validate_instance(result);
    return result;
}
