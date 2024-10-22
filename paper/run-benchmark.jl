using JSOSolvers
using OptimizationProblems
using OptimizationProblems.ADNLPProblems
using ADNLPModels
using SolverBenchmark
using NLPModels
using DataFrames
using CSV

include("paper/run_benchmark_utils.jl")

# Options
β = 0.9 # momentum contribution parameter
atol = 1e-4 # first order criterion absolute tolerance
rtol = 1e-4 # first order criterion relative tolerance
max_eval = -1 # max number of evaluation, -1 -> no limit
max_iter = typemax(Int) # max number of itration
dimensions = [100,1000,2000,5000] # dimension of problem sets
max_time = Dict(100 => 60., 1000 => 60., 2000 => 120., 5000 => 120.) # max time per problem dimension, must match values in dimensions vector
NM = [1,2,5,10] # non-monotone parameter, i.e. number of previous values of the objective memorized. 1 is monotone case.
solvers = [:r2,:tr,:fomo_r2,:fomo_tr] # algorithms being compared

# run solvers on problem set 
# /!\ Might take a while depending on the number of problem sets and alogrithms to be benchmarked
stats = run_benchmark(dimensions, NM, solvers, β, atol, rtol, max_eval, max_iter, max_time::Dict) # run solvers over problems set 

# export data as .csv files. Name format is solvername_dimension_
export_folder = "docs/src/bench_data/" # change this for the folder path where you want to export the data
save_data(export_folder,stats)