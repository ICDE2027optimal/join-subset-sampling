// Measure construction and sampling for the materialized subset index.
#include "core.hpp"
#include "materialized_explicit.hpp"
#include "orient.hpp"
#include "prepared_loader.hpp"

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

constexpr double METADATA_REL_TOLERANCE = 1e-9;
constexpr double METADATA_ABS_TOLERANCE = 1e-15;
constexpr std::uint64_t MAX_VALIDATION_JOIN_SIZE = 100000;
constexpr std::uint64_t MAX_VALIDATION_ROW_EXAMINATIONS = 1000000;
constexpr double VALIDATION_MAX_Z = 7.0;

struct Args {
    std::string query_directory;
    std::string weight_set;
    std::string method;
    std::string case_label;
    int root = -1;
    std::uint64_t seed = 0;
    std::uint64_t warmup = 0;
    std::uint64_t queries = 0;
    std::uint64_t expected_j = 0;
    double expected_mu = 0.0;
    double expected_maxp = 0.0;
    double expected_sump2 = 0.0;
    bool validate_only = false;
};

struct CommonTimes {
    double load_ms = 0.0;
    double orient_ms = 0.0;
};

struct MethodMetadata {
    std::optional<std::uint64_t> join_size;
    std::optional<double> mu;
    std::optional<double> max_probability;
    std::optional<double> sum_probability_squared;
    std::size_t approximate_bytes = 0;
    std::optional<double> candidate_poisson_factor;
    std::optional<int> score_limit;
    std::optional<double> score_quantum;
    std::optional<std::uint64_t> score_convolution_products;
    std::optional<std::uint64_t> score_convolution_calls;
    std::optional<std::uint64_t> score_direct_calls;
    std::optional<std::uint64_t> score_ntt2_calls;
    std::optional<std::uint64_t> score_ntt3_calls;
    std::optional<std::uint64_t> score_direct_pairs;
    std::optional<std::uint64_t> score_transform_points_times_primes;
    std::optional<double> global_envelope_probability;
    std::optional<double> global_envelope_mass;
    std::optional<double> root_envelope_mass;
    std::optional<double> rank_output_probability;
    std::optional<std::uint64_t> scan_tuple_extensions;
    std::optional<std::uint64_t> materialized_relations;
    std::optional<std::uint64_t> materialized_outputs;
    std::optional<std::uint64_t> flat_id_cells;
    std::optional<std::uint64_t> flat_id_capacity;
    std::optional<std::uint64_t> flat_id_bytes;
    std::optional<std::uint64_t> subset_bucket_count;
    std::optional<std::uint64_t> subset_tail_size;
    std::optional<std::uint64_t> subset_index_bytes;
    std::optional<std::uint64_t> subset_dynamic_bytes;
    std::optional<std::uint64_t> enumeration_tuple_extensions;
    std::optional<std::uint64_t> enumeration_outputs;
    std::optional<std::uint64_t> temporary_probability_cells;
    std::optional<std::uint64_t> temporary_probability_capacity;
    std::optional<std::uint64_t> temporary_probability_bytes;
    std::optional<std::uint64_t> owned_persistent_bytes;
    std::optional<std::uint64_t> diagnostic_selected_indices;
    std::optional<std::uint64_t> diagnostic_outputs;
    std::optional<std::uint64_t> diagnostic_selected_index_temporary_bytes;
};

struct TimedBatch {
    double milliseconds = 0.0;
    std::uint64_t sum_output = 0;
};

struct ValidationResult {
    std::uint64_t oracle_join_size = 0;
    std::uint64_t oracle_row_examinations = 0;
    double oracle_mu = 0.0;
    double oracle_max_probability = 0.0;
    double oracle_sum_p2 = 0.0;
    std::uint64_t trials = 0;
    std::uint64_t sum_output = 0;
    double maximum_marginal_z = 0.0;
};

[[noreturn]] void usage_error(const std::string& message) {
    throw std::invalid_argument(
        message +
        "\nusage: materialized_explicit_driver --query-dir DIR --weight-set REL --root R "
        "--method materialized-explicit --seed S --warmup W "
        "--queries Q --expected-j J --expected-mu MU --expected-maxp P "
        "--expected-sump2 S2 [--case-label LABEL] [--validate-only]");
}

std::uint64_t parse_u64(const std::string& value, const char* name) {
    std::uint64_t result = 0;
    const auto parsed = std::from_chars(
        value.data(), value.data() + value.size(), result);
    if (parsed.ec != std::errc() || parsed.ptr != value.data() + value.size())
        usage_error(std::string("invalid ") + name + ": " + value);
    return result;
}

int parse_int(const std::string& value, const char* name) {
    int result = 0;
    const auto parsed = std::from_chars(
        value.data(), value.data() + value.size(), result);
    if (parsed.ec != std::errc() || parsed.ptr != value.data() + value.size())
        usage_error(std::string("invalid ") + name + ": " + value);
    return result;
}

double parse_nonnegative_double(const std::string& value, const char* name) {
    if (value.empty()) usage_error(std::string("empty ") + name);
    errno = 0;
    char* end = nullptr;
    const double result = std::strtod(value.c_str(), &end);
    if (errno == ERANGE || end != value.c_str() + value.size() ||
        !std::isfinite(result) || result < 0.0)
        usage_error(std::string("invalid ") + name + ": " + value);
    return result;
}

Args parse_args(int argc, char** argv) {
    Args args;
    bool have_query_directory = false;
    bool have_weight_set = false;
    bool have_method = false;
    bool have_root = false;
    bool have_seed = false;
    bool have_warmup = false;
    bool have_queries = false;
    bool have_expected_j = false;
    bool have_expected_mu = false;
    bool have_expected_maxp = false;
    bool have_expected_sump2 = false;

    for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];
        if (option == "--validate-only") {
            args.validate_only = true;
            continue;
        }
        if (option == "--help")
            usage_error("materialized explicit join driver");
        if (i + 1 >= argc) usage_error("missing value for " + option);
        const std::string value = argv[++i];
        if (option == "--query-dir") {
            args.query_directory = value;
            have_query_directory = true;
        } else if (option == "--weight-set") {
            args.weight_set = value;
            have_weight_set = true;
        } else if (option == "--root") {
            args.root = parse_int(value, "root");
            have_root = true;
        } else if (option == "--method") {
            args.method = value;
            have_method = true;
        } else if (option == "--seed") {
            args.seed = parse_u64(value, "seed");
            have_seed = true;
        } else if (option == "--warmup") {
            args.warmup = parse_u64(value, "warmup");
            have_warmup = true;
        } else if (option == "--queries") {
            args.queries = parse_u64(value, "queries");
            have_queries = true;
        } else if (option == "--expected-j") {
            args.expected_j = parse_u64(value, "expected-j");
            have_expected_j = true;
        } else if (option == "--expected-mu") {
            args.expected_mu = parse_nonnegative_double(value, "expected-mu");
            have_expected_mu = true;
        } else if (option == "--expected-maxp") {
            args.expected_maxp =
                parse_nonnegative_double(value, "expected-maxp");
            have_expected_maxp = true;
        } else if (option == "--expected-sump2") {
            args.expected_sump2 =
                parse_nonnegative_double(value, "expected-sump2");
            have_expected_sump2 = true;
        } else if (option == "--case-label") {
            args.case_label = value;
        } else {
            usage_error("unknown option: " + option);
        }
    }

    if (!have_query_directory || !have_weight_set || !have_method ||
        !have_root || !have_seed || !have_warmup || !have_queries ||
        !have_expected_j || !have_expected_mu || !have_expected_maxp ||
        !have_expected_sump2)
        usage_error("missing required option");
    if (args.method != "materialized-explicit")
        usage_error("unknown method: " + args.method);
    if (args.queries == 0)
        usage_error("queries must be positive");
    if (args.expected_maxp > 1.0)
        usage_error("expected-maxp must lie in [0,1]");
    return args;
}

double elapsed_ms(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

std::size_t input_tuple_count(const Instance& instance) {
    std::size_t result = 0;
    for (const Relation& relation : instance.rel) {
        if (std::numeric_limits<std::size_t>::max() - result <
            relation.tuples.size())
            throw std::overflow_error("input tuple count overflow");
        result += relation.tuples.size();
    }
    return result;
}

bool close_metadata(double actual, double expected) {
    const double scale = std::max(std::abs(actual), std::abs(expected));
    return std::abs(actual - expected) <=
        METADATA_ABS_TOLERANCE + METADATA_REL_TOLERANCE * scale;
}

void check_method_metadata(const Args& args, const MethodMetadata& metadata) {
    if (metadata.join_size && *metadata.join_size != args.expected_j)
        throw std::runtime_error("method join size disagrees with prepared metadata");
    if (metadata.mu && !close_metadata(*metadata.mu, args.expected_mu))
        throw std::runtime_error("method mu disagrees with prepared metadata");
    if (metadata.max_probability &&
        !close_metadata(*metadata.max_probability, args.expected_maxp))
        throw std::runtime_error(
            "method maximum probability disagrees with prepared metadata");
    if (metadata.sum_probability_squared &&
        !close_metadata(*metadata.sum_probability_squared,
                        args.expected_sump2))
        throw std::runtime_error(
            "method sum p squared disagrees with prepared metadata");
}

std::vector<DrawnOutput> tiny_oracle(
    const Instance& instance, std::uint64_t& row_examinations) {
    std::vector<DrawnOutput> output;
    Output ids(static_cast<std::size_t>(instance.m), -1);
    std::vector<int> chosen(static_cast<std::size_t>(instance.m), -1);
    std::function<void(std::size_t, double)> visit =
        [&](std::size_t depth, double probability) {
            if (depth == instance.preorder.size()) {
                if (output.size() >= MAX_VALIDATION_JOIN_SIZE)
                    throw std::runtime_error(
                        "validate-only join exceeds tiny oracle limit");
                output.push_back(DrawnOutput{ids, probability});
                return;
            }
            const int relation_id = instance.preorder[depth];
            const Relation& relation =
                instance.rel[static_cast<std::size_t>(relation_id)];
            for (std::size_t tuple_index = 0;
                 tuple_index < relation.tuples.size(); ++tuple_index) {
                if (++row_examinations > MAX_VALIDATION_ROW_EXAMINATIONS)
                    throw std::runtime_error(
                        "validate-only oracle exceeds row-examination limit");
                const Tuple& tuple = relation.tuples[tuple_index];
                const int parent =
                    instance.parent[static_cast<std::size_t>(relation_id)];
                if (parent >= 0) {
                    const Tuple& parent_tuple =
                        instance.rel[static_cast<std::size_t>(parent)].tuples[
                            static_cast<std::size_t>(
                                chosen[static_cast<std::size_t>(parent)])];
                    if (tuple.edge_key[static_cast<std::size_t>(parent)] !=
                        parent_tuple.edge_key[
                            static_cast<std::size_t>(relation_id)])
                        continue;
                }
                chosen[static_cast<std::size_t>(relation_id)] =
                    static_cast<int>(tuple_index);
                ids[static_cast<std::size_t>(relation_id)] = tuple.id;
                visit(depth + 1, probability * tuple.weight);
            }
        };
    visit(0, 1.0);
    return output;
}

ValidationResult prepare_oracle(const Args& args, const Instance& instance,
                                std::vector<DrawnOutput>& oracle) {
    ValidationResult result;
    oracle = tiny_oracle(instance, result.oracle_row_examinations);
    result.oracle_join_size = static_cast<std::uint64_t>(oracle.size());
    result.oracle_max_probability = 0.0;
    long double mu = 0.0L;
    long double sum_p2 = 0.0L;
    std::unordered_set<Output, VecHash> identities;
    identities.reserve(oracle.size());
    for (const DrawnOutput& item : oracle) {
        if (!identities.insert(item.ids).second)
            throw std::runtime_error("tiny oracle produced duplicate tuple identity");
        mu += static_cast<long double>(item.p);
        sum_p2 += static_cast<long double>(item.p) *
                  static_cast<long double>(item.p);
        result.oracle_max_probability =
            std::max(result.oracle_max_probability, item.p);
    }
    result.oracle_mu = static_cast<double>(mu);
    result.oracle_sum_p2 = static_cast<double>(sum_p2);
    if (result.oracle_join_size != args.expected_j ||
        !close_metadata(result.oracle_mu, args.expected_mu) ||
        !close_metadata(result.oracle_max_probability, args.expected_maxp) ||
        !close_metadata(result.oracle_sum_p2, args.expected_sump2))
        throw std::runtime_error(
            "tiny oracle disagrees with preparation-provided metadata");
    return result;
}

template<class Sampler>
ValidationResult validate_samples(const Args& args, const Instance& instance,
                                  Sampler&& sample) {
    std::vector<DrawnOutput> oracle;
    ValidationResult result = prepare_oracle(args, instance, oracle);
    result.trials = args.queries;

    std::unordered_map<Output, std::size_t, VecHash> position;
    position.reserve(oracle.size());
    for (std::size_t i = 0; i < oracle.size(); ++i)
        position.emplace(oracle[i].ids, i);
    std::vector<std::uint64_t> observed(oracle.size(), 0);

    for (std::uint64_t trial = 0; trial < args.queries; ++trial) {
        const auto output = sample();
        if (std::numeric_limits<std::uint64_t>::max() - result.sum_output <
            output.size())
            throw std::overflow_error("validation output count overflow");
        result.sum_output += static_cast<std::uint64_t>(output.size());
        std::unordered_set<Output, VecHash> unique;
        unique.reserve(output.size());
        for (const Output& ids : output) {
            if (!unique.insert(ids).second)
                throw std::runtime_error(
                    "sample contains duplicate full tuple identity");
            const auto found = position.find(ids);
            if (found == position.end())
                throw std::runtime_error("sample contains a non-join output");
            ++observed[found->second];
        }
    }

    for (std::size_t i = 0; i < oracle.size(); ++i) {
        const double probability = oracle[i].p;
        const double expected =
            static_cast<double>(args.queries) * probability;
        const double variance =
            static_cast<double>(args.queries) * probability *
            (1.0 - probability);
        if (variance == 0.0) {
            const std::uint64_t exact = probability >= 1.0 ? args.queries : 0;
            if (observed[i] != exact)
                throw std::runtime_error(
                    "deterministic validation marginal disagrees with oracle");
            continue;
        }
        const double z =
            std::abs(static_cast<double>(observed[i]) - expected) /
            std::sqrt(variance);
        result.maximum_marginal_z =
            std::max(result.maximum_marginal_z, z);
    }
    if (result.maximum_marginal_z > VALIDATION_MAX_Z)
        throw std::runtime_error("validation marginal exceeds declared z limit");
    return result;
}

template<class Sampler>
TimedBatch time_samples(std::uint64_t count, Sampler&& sample) {
    TimedBatch result;
    const auto begin = Clock::now();
    for (std::uint64_t query = 0; query < count; ++query) {
        std::size_t output_size = 0;
        {
            auto output = sample();
            output_size = output.size();
        }  
        if (std::numeric_limits<std::uint64_t>::max() - result.sum_output <
            output_size)
            throw std::overflow_error("timed output count overflow");
        result.sum_output += static_cast<std::uint64_t>(output_size);
    }
    const auto end = Clock::now();
    result.milliseconds = elapsed_ms(begin, end);
    return result;
}

std::string json_escape(const std::string& value) {
    std::ostringstream output;
    for (unsigned char c : value) {
        switch (c) {
            case '\\': output << "\\\\"; break;
            case '"': output << "\\\""; break;
            case '\b': output << "\\b"; break;
            case '\f': output << "\\f"; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default:
                if (c < 0x20) {
                    output << "\\u" << std::hex << std::setw(4)
                           << std::setfill('0') << static_cast<unsigned>(c)
                           << std::dec << std::setfill(' ');
                } else {
                    output << static_cast<char>(c);
                }
        }
    }
    return output.str();
}

void emit_optional_u64(const std::optional<std::uint64_t>& value) {
    if (value) std::cout << *value;
    else std::cout << "null";
}

void emit_optional_double(const std::optional<double>& value) {
    if (value) std::cout << *value;
    else std::cout << "null";
}

void emit_optional_int(const std::optional<int>& value) {
    if (value) std::cout << *value;
    else std::cout << "null";
}

void emit_record(const Args& args, const Instance& instance,
                 const CommonTimes& common, double build_ms,
                 const MethodMetadata& metadata,
                 const std::optional<TimedBatch>& warmup,
                 const std::optional<TimedBatch>& measured,
                 const std::optional<ValidationResult>& validation) {
    std::cout << std::setprecision(17);
    std::cout << '{'
              << "\"schema\":\"join-formal-driver-v1\","
              << "\"mode\":\""
              << (args.validate_only ? "validation" : "measurement") << "\","
              << "\"timing_is_benchmark\":"
              << (args.validate_only ? "false" : "true") << ','
              << "\"case_label\":\"" << json_escape(args.case_label) << "\","
              << "\"query_directory\":\""
              << json_escape(args.query_directory) << "\","
              << "\"weight_set\":\"" << json_escape(args.weight_set) << "\","
              << "\"root\":" << args.root << ','
              << "\"method\":\"" << args.method << "\","
              << "\"seed\":" << args.seed << ','
              << "\"warmup_queries\":" << args.warmup << ','
              << "\"measured_queries\":" << args.queries << ','
              << "\"m\":" << instance.m << ','
              << "\"input_tuples\":" << input_tuple_count(instance) << ','
              << "\"prepared_expected_j\":" << args.expected_j << ','
              << "\"prepared_expected_mu\":" << args.expected_mu << ','
              << "\"prepared_expected_maxp\":" << args.expected_maxp << ','
              << "\"prepared_expected_sump2\":" << args.expected_sump2 << ','
              << "\"expected_output_per_query\":" << args.expected_mu << ','
              << "\"metadata_rel_tolerance\":"
              << METADATA_REL_TOLERANCE << ','
              << "\"metadata_abs_tolerance\":"
              << METADATA_ABS_TOLERANCE << ','
              << "\"method_join_size\":";
    emit_optional_u64(metadata.join_size);
    std::cout << ",\"method_mu\":";
    emit_optional_double(metadata.mu);
    std::cout << ",\"method_maxp\":";
    emit_optional_double(metadata.max_probability);
    std::cout << ",\"method_sump2\":";
    emit_optional_double(metadata.sum_probability_squared);
    std::cout << ",\"index_bytes_est\":" << metadata.approximate_bytes
              << ",\"index_bytes_kind\":\"owned_approximate\""
              << ",\"method_diagnostics\":{"
              << "\"candidate_poisson_factor\":";
    emit_optional_double(metadata.candidate_poisson_factor);
    std::cout << ",\"score_L\":";
    emit_optional_int(metadata.score_limit);
    std::cout << ",\"score_h\":";
    emit_optional_double(metadata.score_quantum);
    std::cout << ",\"score_convolution_products\":";
    emit_optional_u64(metadata.score_convolution_products);
    std::cout << ",\"score_convolution_calls\":";
    emit_optional_u64(metadata.score_convolution_calls);
    std::cout << ",\"score_direct_calls\":";
    emit_optional_u64(metadata.score_direct_calls);
    std::cout << ",\"score_ntt2_calls\":";
    emit_optional_u64(metadata.score_ntt2_calls);
    std::cout << ",\"score_ntt3_calls\":";
    emit_optional_u64(metadata.score_ntt3_calls);
    std::cout << ",\"score_direct_pairs\":";
    emit_optional_u64(metadata.score_direct_pairs);
    std::cout << ",\"score_transform_points_times_primes\":";
    emit_optional_u64(metadata.score_transform_points_times_primes);
    std::cout << ",\"global_envelope_probability\":";
    emit_optional_double(metadata.global_envelope_probability);
    std::cout << ",\"global_envelope_mass\":";
    emit_optional_double(metadata.global_envelope_mass);
    std::cout << ",\"root_envelope_mass\":";
    emit_optional_double(metadata.root_envelope_mass);
    std::cout << ",\"rank_output_probability\":";
    emit_optional_double(metadata.rank_output_probability);
    std::cout << ",\"scan_tuple_extensions\":";
    emit_optional_u64(metadata.scan_tuple_extensions);
    std::cout << ",\"materialized_relations\":";
    emit_optional_u64(metadata.materialized_relations);
    std::cout << ",\"materialized_outputs\":";
    emit_optional_u64(metadata.materialized_outputs);
    std::cout << ",\"flat_id_cells\":";
    emit_optional_u64(metadata.flat_id_cells);
    std::cout << ",\"flat_id_capacity\":";
    emit_optional_u64(metadata.flat_id_capacity);
    std::cout << ",\"flat_id_bytes\":";
    emit_optional_u64(metadata.flat_id_bytes);
    std::cout << ",\"subset_bucket_count\":";
    emit_optional_u64(metadata.subset_bucket_count);
    std::cout << ",\"subset_tail_size\":";
    emit_optional_u64(metadata.subset_tail_size);
    std::cout << ",\"subset_index_bytes\":";
    emit_optional_u64(metadata.subset_index_bytes);
    std::cout << ",\"subset_dynamic_bytes\":";
    emit_optional_u64(metadata.subset_dynamic_bytes);
    std::cout << ",\"enumeration_tuple_extensions\":";
    emit_optional_u64(metadata.enumeration_tuple_extensions);
    std::cout << ",\"enumeration_outputs\":";
    emit_optional_u64(metadata.enumeration_outputs);
    std::cout << ",\"temporary_probability_cells\":";
    emit_optional_u64(metadata.temporary_probability_cells);
    std::cout << ",\"temporary_probability_capacity\":";
    emit_optional_u64(metadata.temporary_probability_capacity);
    std::cout << ",\"temporary_probability_bytes\":";
    emit_optional_u64(metadata.temporary_probability_bytes);
    std::cout << ",\"owned_persistent_bytes\":";
    emit_optional_u64(metadata.owned_persistent_bytes);
    std::cout << ",\"diagnostic_selected_indices\":";
    emit_optional_u64(metadata.diagnostic_selected_indices);
    std::cout << ",\"diagnostic_outputs\":";
    emit_optional_u64(metadata.diagnostic_outputs);
    std::cout << ",\"diagnostic_selected_index_temporary_bytes\":";
    emit_optional_u64(metadata.diagnostic_selected_index_temporary_bytes);
    std::cout << ",\"diagnostic_query_scope\":\""
              << (args.validate_only
                      ? "post-validation-untimed-copied-rng-probe"
                      : "post-measurement-untimed-copied-rng-probe")
              << '"';
    std::cout << '}'
              << ",\"load_ms\":" << common.load_ms
              << ",\"orient_ms\":" << common.orient_ms
              << ",\"method_build_ms\":" << build_ms;
    if (warmup) {
        std::cout << ",\"warmup_ms\":" << warmup->milliseconds
                  << ",\"warmup_sum_output\":" << warmup->sum_output;
    } else {
        std::cout << ",\"warmup_ms\":null,\"warmup_sum_output\":null";
    }
    if (measured) {
        std::cout << ",\"query_ms_total\":" << measured->milliseconds
                  << ",\"query_ms_mean\":"
                  << measured->milliseconds /
                         static_cast<double>(args.queries)
                  << ",\"sum_output\":" << measured->sum_output;
    } else {
        std::cout << ",\"query_ms_total\":null,\"query_ms_mean\":null,"
                     "\"sum_output\":null";
    }
    std::cout << ",\"rss_external_kib\":null"
              << ",\"rss_measurement\":\"external-process-wrapper\""
              << ",\"common_planning_included\":false"
              << ",\"preparation_metadata_external\":true";
    if (validation) {
        std::cout << ",\"validation\":{"
                  << "\"oracle_join_size\":"
                  << validation->oracle_join_size << ','
                  << "\"oracle_row_examinations\":"
                  << validation->oracle_row_examinations << ','
                  << "\"oracle_mu\":" << validation->oracle_mu << ','
                  << "\"oracle_maxp\":"
                  << validation->oracle_max_probability << ','
                  << "\"oracle_sump2\":"
                  << validation->oracle_sum_p2 << ','
                  << "\"trials\":" << validation->trials << ','
                  << "\"sum_output\":" << validation->sum_output << ','
                  << "\"maximum_marginal_z\":"
                  << validation->maximum_marginal_z << ','
                  << "\"maximum_allowed_z\":" << VALIDATION_MAX_Z
                  << '}';
    } else {
        std::cout << ",\"validation\":null";
    }
    std::cout << "}\n";
}

template<class Sampler, class PostQueries>
int finish_method(const Args& args, const Instance& instance,
                  const CommonTimes& common, double build_ms,
                  MethodMetadata& metadata, Sampler&& sample,
                  PostQueries&& post_queries) {
    check_method_metadata(args, metadata);
    if (args.validate_only) {
        const ValidationResult validation =
            validate_samples(args, instance, sample);
        post_queries();
        emit_record(args, instance, common, build_ms, metadata,
                    std::nullopt, std::nullopt, validation);
        return 0;
    }
    const TimedBatch warmup = time_samples(args.warmup, sample);
    const TimedBatch measured = time_samples(args.queries, sample);
    post_queries();
    emit_record(args, instance, common, build_ms, metadata,
                warmup, measured, std::nullopt);
    return 0;
}

int run_method(const Args& args, const Instance& instance,
               const CommonTimes& common) {
    std::mt19937_64 rng(args.seed);
    const auto begin = Clock::now();
    auto index = std::make_unique<
        materialized_explicit::MaterializedExplicitSampler>(instance);
    const auto end = Clock::now();

    MethodMetadata metadata;
    metadata.join_size = index->join_size();
    metadata.mu = static_cast<double>(index->mu());
    metadata.max_probability = index->max_probability();
    metadata.sum_probability_squared =
        static_cast<double>(index->sum_p2());
    metadata.approximate_bytes = index->approximate_bytes();
    metadata.materialized_relations =
        static_cast<std::uint64_t>(index->relation_count());
    metadata.subset_bucket_count =
        static_cast<std::uint64_t>(index->bucket_count());
    metadata.subset_tail_size =
        static_cast<std::uint64_t>(index->tail_size());

    const auto& build_stats = index->build_stats();
    if (build_stats.materialized_outputs != index->join_size() ||
        build_stats.relations != index->relation_count() ||
        build_stats.owned_persistent_bytes != index->approximate_bytes())
        throw std::logic_error(
            "materialized explicit accessor/build metadata mismatch");
    metadata.materialized_outputs = build_stats.materialized_outputs;
    metadata.flat_id_cells =
        static_cast<std::uint64_t>(build_stats.flat_id_cells);
    metadata.flat_id_capacity =
        static_cast<std::uint64_t>(build_stats.flat_id_capacity);
    metadata.flat_id_bytes =
        static_cast<std::uint64_t>(build_stats.flat_id_bytes);
    metadata.subset_index_bytes =
        static_cast<std::uint64_t>(build_stats.subset_index_bytes);
    metadata.subset_dynamic_bytes =
        static_cast<std::uint64_t>(build_stats.subset_dynamic_bytes);
    metadata.enumeration_tuple_extensions =
        build_stats.enumeration.tuple_extensions;
    metadata.enumeration_outputs = build_stats.enumeration.outputs;
    metadata.temporary_probability_cells = static_cast<std::uint64_t>(
        build_stats.temporary_probability_cells);
    metadata.temporary_probability_capacity = static_cast<std::uint64_t>(
        build_stats.temporary_probability_capacity);
    metadata.temporary_probability_bytes = static_cast<std::uint64_t>(
        build_stats.temporary_probability_bytes);
    metadata.owned_persistent_bytes = static_cast<std::uint64_t>(
        build_stats.owned_persistent_bytes);

    auto sample = [&]() { return index->sample_vector(rng); };

    auto post_queries = [&]() {
        std::mt19937_64 diagnostic_rng = rng;
        materialized_explicit::QueryStats diagnostic_stats;
        {
            const auto diagnostic_output =
                index->sample_vector(diagnostic_rng, diagnostic_stats);
            metadata.diagnostic_outputs =
                static_cast<std::uint64_t>(diagnostic_output.size());
        }
        metadata.diagnostic_selected_indices =
            static_cast<std::uint64_t>(diagnostic_stats.selected_indices);
        metadata.diagnostic_selected_index_temporary_bytes =
            static_cast<std::uint64_t>(
                diagnostic_stats.selected_index_temporary_bytes);
    };
    return finish_method(args, instance, common, elapsed_ms(begin, end),
                         metadata, sample, post_queries);
}
}  

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        CommonTimes common;
        const auto load_begin = Clock::now();
        UndirectedInstance undirected = load_prepared_instance(
            args.query_directory, args.weight_set);
        const auto load_end = Clock::now();
        common.load_ms = elapsed_ms(load_begin, load_end);

        const auto orient_begin = Clock::now();
        Instance instance = orient_instance(undirected, args.root);
        const auto orient_end = Clock::now();
        common.orient_ms = elapsed_ms(orient_begin, orient_end);
        return run_method(args, instance, common);
    } catch (const std::exception& error) {
        std::cerr << "materialized_explicit_driver: " << error.what() << '\n';
        return 2;
    }
}
