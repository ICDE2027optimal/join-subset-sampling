#include "orient.hpp"
#include "prepared_loader.hpp"

#include <charconv>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr const char* usage =
    "Usage: product_sample --query-dir PATH --weights RELATIVE_DIR "
    "--root N --seed UINT64 --calls POSITIVE_UINT64\n"
    "Read a stats-undirected-v1 directory and emit one sampled subset per JSON line.\n"
    "Each line contains call (zero-based), mu, and sample (tuple-ID vectors in relation order).\n"
    "Tuple-ID vectors are sorted lexicographically within each subset.\n";

struct Options {
    std::string query_directory;
    std::string weights;
    int root = 0;
    std::uint64_t seed = 0;
    std::uint64_t calls = 0;
};

template<class Integer>
Integer parse_unsigned(std::string_view value, const std::string& name) {
    Integer result = 0;
    if (value.empty() || value.front() < '0' || value.front() > '9')
        throw std::invalid_argument(name + " requires a decimal unsigned integer");
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    if (parsed.ec != std::errc() || parsed.ptr != value.data() + value.size())
        throw std::invalid_argument(name + " requires a decimal unsigned integer in range");
    return result;
}

Options parse_options(int argc, char** argv) {
    Options options;
    unsigned seen = 0;
    for (int argument = 1; argument < argc; ++argument) {
        const std::string name(argv[argument]);
        unsigned bit = 0;
        if (name == "--query-dir") bit = 1;
        else if (name == "--weights") bit = 2;
        else if (name == "--root") bit = 4;
        else if (name == "--seed") bit = 8;
        else if (name == "--calls") bit = 16;
        else throw std::invalid_argument("unknown option: " + name);
        if ((seen & bit) != 0)
            throw std::invalid_argument("duplicate option: " + name);
        if (++argument == argc || std::string_view(argv[argument]).starts_with("--"))
            throw std::invalid_argument("missing value for " + name);
        const std::string value(argv[argument]);
        if (value.empty()) throw std::invalid_argument("empty value for " + name);
        seen |= bit;
        if (bit == 1) options.query_directory = value;
        else if (bit == 2) options.weights = value;
        else if (bit == 4) {
            const auto root = parse_unsigned<std::uint64_t>(value, name);
            if (root > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
                throw std::invalid_argument("--root exceeds the supported relation-index range");
            options.root = static_cast<int>(root);
        } else if (bit == 8) options.seed = parse_unsigned<std::uint64_t>(value, name);
        else options.calls = parse_unsigned<std::uint64_t>(value, name);
    }
    if (seen != 31)
        throw std::invalid_argument("--query-dir, --weights, --root, --seed and --calls are required");
    if (options.calls == 0) throw std::invalid_argument("--calls must be positive");
    const std::filesystem::path weights(options.weights);
    if (weights.has_root_name() || weights.has_root_directory())
        throw std::invalid_argument("--weights must be relative to --query-dir");
    for (const auto& component : weights)
        if (component == "..")
            throw std::invalid_argument("--weights cannot contain a parent-directory component");
    return options;
}

void write_sample(std::uint64_t call, double mu,
                  const std::unordered_set<Output, VecHash>& subset) {
    std::vector<Output> ordered(subset.begin(), subset.end());
    std::sort(ordered.begin(), ordered.end());
    std::cout << "{\"call\":" << call << ",\"mu\":" << mu << ",\"sample\":[";
    bool first_output = true;
    for (const auto& ids : ordered) {
        if (!first_output) std::cout << ',';
        first_output = false;
        std::cout << '[';
        for (std::size_t relation = 0; relation < ids.size(); ++relation) {
            if (relation != 0) std::cout << ',';
            std::cout << ids[relation];
        }
        std::cout << ']';
    }
    std::cout << "]}\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::cout.exceptions(std::ios::badbit | std::ios::failbit);
        if (argc == 2 && (std::string_view(argv[1]) == "--help" ||
                          std::string_view(argv[1]) == "-h")) {
            std::cout << usage;
            return 0;
        }
        const Options options = parse_options(argc, argv);
        const UndirectedInstance input =
            load_prepared_instance(options.query_directory, options.weights);
        const Instance instance = orient_instance(input, options.root);
        const WeightedJoinIndex index(instance);
        std::mt19937_64 random(options.seed);
        std::cout << std::setprecision(std::numeric_limits<double>::max_digits10);
        for (std::uint64_t call = 0; call < options.calls; ++call) {
            QueryStats stats;
            const auto subset = sample_subset(index, random, stats);
            write_sample(call, index.mu(), subset);
        }
        std::cout.flush();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "product_sample: " << error.what() << '\n';
        return 2;
    }
}
