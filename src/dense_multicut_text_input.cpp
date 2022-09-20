#include "dense_gaec.h"
#include "dense_gaec_parallel.h"
#include "dense_gaec_adj_matrix.h"
#include "dense_features_parser.h"
#include <iostream>
#include <functional>
#include <CLI/CLI.hpp>

using namespace DENSE_MULTICUT;

int main(int argc, char** argv)
{
    CLI::App app("Dense multicut solvers");
    std::vector<std::string> available_solvers{"adj_matrix","flat_index", "hnsw", "parallel_flat_index", "parallel_hnsw"};

    std::string file_path, solver_type;
    app.add_option("-f,--file,file_pos", file_path, "Path to dense multicut instance (.txt)")->required()->check(CLI::ExistingPath);
    app.add_option("-s,--solver,solver_pos", solver_type, "One of the following solver types: \n"
    "adj_matrix\n, flat_index\n, hnsw\n, parallel_flat_index\n, parallel_hnsw")->required();

    app.parse(argc, argv);
    size_t num_nodes, dim;
    std::vector<float> features;

    std::tie(features, num_nodes, dim) = read_file(file_path);
    if (solver_type ==  "adj_matrix")
        dense_gaec_adj_matrix(num_nodes, dim, features);
    else if (solver_type ==  "flat_index")
        dense_gaec_flat_index(num_nodes, dim, features);
    else if (solver_type ==  "hnsw")
        dense_gaec_hnsw(num_nodes, dim, features);
    else if (solver_type ==  "parallel_flat_index")
        dense_gaec_parallel_flat_index(num_nodes, dim, features);
    else if (solver_type ==  "parallel_hnsw")
        dense_gaec_parallel_hnsw(num_nodes, dim, features);
    else if (solver_type ==  "flat_index")
        dense_gaec_flat_index(num_nodes, dim, features);
    else
        throw std::runtime_error("Unknown solver type: " + solver_type);
}