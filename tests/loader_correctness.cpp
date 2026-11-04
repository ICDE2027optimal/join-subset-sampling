// Validation helpers for the join-sampling implementation.
#include "prepared_loader.hpp"
#include "orient.hpp"
#include <iostream>
#include <set>

int main(int argc,char** argv) {
    try {
        if (argc!=2) throw std::invalid_argument("pass the tiny fixture directory");
        auto input=load_prepared_instance(argv[1],"weights/uniform/scale-1");
        const std::set<Output> expected{{10,20,30,40},{10,21,31,40},{10,21,32,41},
                                      {11,22,31,40},{11,22,32,41}};
        for (int root=0;root<input.m;++root) {
            auto directed=orient_instance(input,root);
            WeightedJoinIndex index(directed);
            if (index.mu()!=5 || index.max_probability()!=1)
                throw std::runtime_error("fixture mass mismatch");
            std::set<Output> observed;
            HeavyStats hs;
            index.enumerate_heavy([&](const Output& ids,double p) {
                if (p!=1) throw std::runtime_error("fixture weight changed");
                observed.insert(ids);
            },hs);
            if (observed!=expected) throw std::runtime_error("fixture identity mismatch");
        }
        auto reject=[](auto operation) {
            try { operation(); } catch (const std::exception&) { return; }
            throw std::runtime_error("malformed prepared value accepted");
        };
        reject([&] {load_prepared_instance(argv[1],"../outside");});
        for (const std::string x : {"", "nan", "inf", "-0.1", "1.01", "0.5garbage"})
            reject([&] {prepared_detail::probability(x);});
        for (const std::string x : {"", "7x", "9223372036854775808"})
            reject([&] {prepared_detail::integer(x);});
        if (prepared_detail::probability("0.125")!=.125 ||
            prepared_detail::integer("-9223372036854775808")!=std::numeric_limits<std::int64_t>::min())
            throw std::runtime_error("valid boundary field rejected");
        std::cout<<"PASS split TSV load, all roots and five exact witnesses; invalid numbers/paths\n";
    } catch (const std::exception& e) {
        std::cerr<<"FAIL "<<e.what()<<'\n'; return 1;
    }
}
