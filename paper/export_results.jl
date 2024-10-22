using SolverBenchmark
using Plots
using DataFrames
using CSV
using PrettyTables
using BenchmarkProfiles
using Statistics

include("paper/export_results_utils.jl")

###### define paths ######
data_folder = "paper/benchmark_data/" # data folder
profiles_folder = "paper/figures/" # perfomance profile figures destination folder 
tables_folder = "paper/tables/" # average beta max tables destination folder
histogram_folder = "paper/histograms_data/" # histograms destination folder
ext = ".pdf" # figure format extension

###### load the data #####
dimensions = [100, 1000, 2000, 5000]
NM = [1, 2, 5, 10]
solvers = [:tr, :fomo_tr, :r2, :fomo_r2]
stats = load_fomo_data(data_folder, dimensions, NM, solvers)

# ###### display the results (table) #####
# # show solvers execution stats as tables for the problem set of dimension dim_table

# # table columns
# cols = [:id, :name, :nvar, :objective, :dual_feas, :neval_obj, :neval_grad, :neval_hess, :iter, :elapsed_time, :status, :avgsatβ]
# header = Dict(
#   :nvar => "n",
#   :objective => "f(x)",
#   :dual_feas => "‖∇f(x)‖",
#   :neval_obj => "# f",
#   :neval_grad => "# ∇f",
#   :elapsed_time => "t",
#   :avgsatβ => "avgβmax"
# )
# dim = 100 # change this value for any element of dimensions (100,1000,2000,5000)
# M = 1
# for solver ∈ solvers
#   println("Problems dimension: $dim - Solver:$(solver)")
#   pretty_stats(stats[dim][M][solver][!, cols], hdr_override=header)
# end


###### export table with names and avgβmax for fomo in the monotone case ########
dimensions = [100, 1000, 2000, 5000] # problem dimensions to be exported
solvers = [:fomo_r2, :fomo_tr]
export_avg_beta_table(stats, solvers, dimensions, tables_folder)


###### export perfomance profiles FOMO monotonous vs standard ######
dimensions = [100, 1000, 2000, 5000] # export profiles for these dimensions
NM = [1] # compare only for M=1 (monotone case)

# plot options
lgd = Dict(100 => true, 1000 => false, 2000 => false, 5000 => false) # show legend for dimension set to true
lgd_pos = [5.0, 4.0] # legend position on the figure
lgd_box_length = 4.0 # legend box length
colours = ["blue", "red"] # plot colours, must match number of algo being compared
styles = ["dashed", "solid"] # plot styles, must match number of algo being compared

# define cost
first_order(df) = df.status .== "first_order"
unbounded(df) = df.status .== "unbounded"
solved(df) = first_order(df) .| unbounded(df)
cost(df) = .!solved(df) .* Inf .+ df.elapsed_time # compare perfomances with respect to solve time

# compare fomo_r2 with r2 in monotone case
solvers = [:fomo_r2, :r2]
solvernames = Dict(:fomo_r2 => "FOMO(R2)", :r2 => "R2")
export_performance_profile(
  stats,
  solvers,
  dimensions,
  NM,
  cost,
  profiles_folder,
  solvernames;
  lgd = lgd,
  lgd_pos = lgd_pos,
  lgd_box_length = lgd_box_length,
  colours = colours,
  linestyles = styles,
)

# compare fomo_tr with tr in monotone case
solvers = [:fomo_tr, :tr]
solvernames = Dict(:fomo_tr => "FOMO(TR)", :tr => "TR")
export_performance_profile(
  stats,
  solvers,
  dimensions,
  NM,
  cost,
  profiles_folder,
  solvernames;
  lgd = lgd,
  lgd_pos = lgd_pos,
  lgd_box_length = lgd_box_length,
  colours = colours,
  linestyles = styles,
)


###### export non-monotonous results: fomo_r2 and fomo_tr for M =1,2,5,10
# plot options
lgd = Dict(100 => true, 1000 => false, 2000 => false, 5000 => false) # show legend for dimension set to true
lgd_pos = [5.0, 4.0] # legend position on the figure
lgd_box_length = 4.0 # legend box length
#colours = ["blue", "red", "black", "green"] # plot colours, must match number of algo being compared
styles = ["solid", "dashed", "dash dot", "dotted"] # plot styles, must match number of algo being compared

# define cost
first_order(df) = df.status .== "first_order"
unbounded(df) = df.status .== "unbounded"
solved(df) = first_order(df) .| unbounded(df)
cost(df) = .!solved(df) .* Inf .+ df.elapsed_time

dimensions = [100, 1000, 2000, 5000]
NM = [1, 2, 5, 10]
solve_tol = 0.05 # solve tolerance: problem is considered solved if objective is within solve_tol*100 % gap with best solution among M=1,2,5,10 

## export non-monotonous fomo_r2 performance profile
solvers = [:fomo_r2]
stats_fomo_r2 = load_fomo_data(data_folder, dimensions, NM, solvers)
set_status_nomin(stats_fomo_r2, solve_tol) 
solvernames = Dict(:fomo_r2 => "")
file_name = "fomo_r2_nonmonotonous"
export_performance_profile(
  stats_fomo_r2,
  solvers,
  dimensions,
  NM,
  cost,
  profiles_folder,
  solvernames;
  file_name = file_name,
  lgd = lgd,
  lgd_pos = lgd_pos,
  lgd_box_length = lgd_box_length,
  #colours = colours,
  linestyles = styles,
)

# export non-monotonous fomo_tr performance profiles
solvers = [:fomo_tr]
stats_fomo_tr = load_fomo_data(data_folder, dimensions, NM, solvers)
set_status_nomin(stats_fomo_tr, solve_tol) 
solvernames = Dict(:fomo_tr => "") # will be displayed in the legend
file_name = "fomo_tr_nonmonotonous"
export_performance_profile(
  stats_fomo_tr,
  solvers,
  dimensions,
  NM,
  cost,
  profiles_folder,
  solvernames;
  file_name = file_name,
  lgd = lgd,
  lgd_pos = lgd_pos,
  lgd_box_length = lgd_box_length,
  #colours = colours,
  linestyles = styles,
)

# generate histograms and average betamax value data
bins = 0:0.2:1.0
solvers = [:fomo_r2, :fomo_tr]
dimensions = [100,1000,2000,5000]
export_histogram_data(stats, solvers, dimensions, bins, histogram_folder)
