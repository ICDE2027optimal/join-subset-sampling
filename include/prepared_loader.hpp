// Read the stats-undirected-v1 input format.
#pragma once

#include "core.hpp"
#include <charconv>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace prepared_detail {
inline std::vector<std::string> fields(std::string line) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    std::vector<std::string> result;
    std::size_t start = 0;
    for (;;) {
        const std::size_t end = line.find('\t', start);
        result.push_back(line.substr(start, end - start));
        if (end == std::string::npos) break;
        start = end + 1;
    }
    return result;
}
inline std::int64_t integer(const std::string& value) {
    std::int64_t answer = 0;
    const auto parsed = std::from_chars(value.data(), value.data()+value.size(), answer);
    if (parsed.ec != std::errc() || parsed.ptr != value.data()+value.size())
        throw std::invalid_argument("invalid prepared integer: " + value);
    return answer;
}
inline int narrow_int(const std::string& value, bool nonnegative = true) {
    const auto answer = integer(value);
    if (answer < (nonnegative ? 0 : std::numeric_limits<int>::min()) ||
        answer > std::numeric_limits<int>::max())
        throw std::invalid_argument("prepared integer outside supported range");
    return static_cast<int>(answer);
}
inline double probability(const std::string& value) {
    if (value.empty()) throw std::invalid_argument("empty prepared weight");
    char* end = nullptr;
    const double result = std::strtod(value.c_str(), &end);
    if (end != value.c_str()+value.size() || !std::isfinite(result) || result<0 || result>1)
        throw std::invalid_argument("invalid prepared weight: " + value);
    return result;
}
inline std::filesystem::path local_path(const std::filesystem::path& root,
                                        const std::string& relative) {
    const std::filesystem::path suffix(relative);
    if (suffix.empty() || suffix.is_absolute())
        throw std::invalid_argument("prepared path must be relative");
    for (const auto& item : suffix)
        if (item == "..") throw std::invalid_argument("prepared path cannot contain ..");
    return root / suffix;
}
inline std::ifstream open(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open prepared file " + path.string());
    return input;
}
inline std::vector<std::string> required_line(std::ifstream& stream) {
    std::string line;
    if (!std::getline(stream,line)) throw std::invalid_argument("truncated prepared file");
    return fields(line);
}
inline void require_end(std::ifstream& stream) {
    std::string line;
    while (std::getline(stream,line))
        if (!line.empty() && line != "\r")
            throw std::invalid_argument("extra prepared rows beyond declared count");
    if (stream.bad()) throw std::runtime_error("prepared file read failed");
}
struct RelationFile { int id; std::size_t rows; std::string keys; };
} 

inline UndirectedInstance load_prepared_instance(
    const std::string& query_directory, const std::string& relative_weight_set) {
    using namespace prepared_detail;
    const std::filesystem::path directory(query_directory);
    const auto weight_directory = local_path(directory,relative_weight_set);
    auto structure = open(directory / "structure.tsv");
    if (required_line(structure) != std::vector<std::string>{"stats-undirected-v1"})
        throw std::invalid_argument("unsupported prepared format");
    UndirectedInstance result;
    result.weight_mode = relative_weight_set;
    std::vector<RelationFile> relation_files;
    bool have_m=false, have_query=false, have_root=false;
    int verification_root=-1;
    std::string line;
    while (std::getline(structure,line)) {
        if (line.empty() || line=="\r") continue;
        const auto f = fields(line);
        if (f[0]=="query" && f.size()==2 && !have_query) {
            result.shape=f[1]; have_query=true;
        } else if (f[0]=="m" && f.size()==2 && !have_m) {
            result.m=narrow_int(f[1]); have_m=true;
        } else if (f[0]=="verification_root" && f.size()==2 && !have_root) {
            verification_root=narrow_int(f[1]); have_root=true;
        } else if (f[0]=="relation" && f.size()==7) {
            relation_files.push_back({narrow_int(f[1]),
                static_cast<std::size_t>(narrow_int(f[4])),f[5]});
        } else if (f[0]=="edge" && f.size()==7) {
            if (integer(f[4])<0) throw std::invalid_argument("negative key cardinality");
            result.edges.push_back({narrow_int(f[1]),narrow_int(f[2]),narrow_int(f[3])});
        } else throw std::invalid_argument("invalid or duplicate prepared structure record");
    }
    if (structure.bad()) throw std::runtime_error("structure read failed");
    if (!have_m || !have_query || !have_root || result.m<=0 ||
        verification_root>=result.m || relation_files.size()!=static_cast<std::size_t>(result.m) ||
        result.edges.size()!=static_cast<std::size_t>(result.m-1))
        throw std::invalid_argument("incomplete prepared structure");
    result.rel.resize(static_cast<std::size_t>(result.m));
    std::vector<unsigned char> seen(result.m,0);
    std::vector<std::vector<EdgeId>> incident(result.m);
    std::unordered_set<EdgeId> edge_ids;
    for (const auto& e : result.edges) {
        if (e.left_relation>=result.m || e.right_relation>=result.m ||
            e.left_relation==e.right_relation || !edge_ids.insert(e.id).second)
            throw std::invalid_argument("invalid prepared edge");
        incident[e.left_relation].push_back(e.id);
        incident[e.right_relation].push_back(e.id);
    }
    for (auto& ids : incident) std::sort(ids.begin(),ids.end());
    for (const auto& descriptor : relation_files) {
        const int v=descriptor.id;
        if (v>=result.m || seen[v]) throw std::invalid_argument("duplicate/invalid relation id");
        seen[v]=1;
        auto keys=open(local_path(directory,descriptor.keys));
        auto weights=open(weight_directory / ("r"+std::to_string(v)+".tsv"));
        std::vector<std::string> key_header{"tuple_id"};
        for (EdgeId e : incident[v]) key_header.push_back("edge_"+std::to_string(e)+"_key");
        if (required_line(keys)!=key_header ||
            required_line(weights)!=std::vector<std::string>{"tuple_id","weight"})
            throw std::invalid_argument("prepared key/weight header mismatch");
        auto& rows=result.rel[v].tuples;
        rows.reserve(descriptor.rows);
        for (std::size_t i=0;i<descriptor.rows;++i) {
            const auto k=required_line(keys), w=required_line(weights);
            if (k.size()!=key_header.size() || w.size()!=2)
                throw std::invalid_argument("prepared row width mismatch");
            const int id=narrow_int(k[0],false);
            if (narrow_int(w[0],false)!=id)
                throw std::invalid_argument("key/weight row identity mismatch");
            UndirectedTuple row;
            row.id=id; row.weight=probability(w[1]);
            for (std::size_t j=0;j<incident[v].size();++j)
                row.incident_keys.emplace_back(incident[v][j],integer(k[j+1]));
            rows.push_back(std::move(row));
        }
        require_end(keys); require_end(weights);
    }

    return result;
}
