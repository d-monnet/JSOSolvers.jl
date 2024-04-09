using JSOSolvers
using OptimizationProblems
using OptimizationProblems.ADNLPProblems
using ADNLPModels
using SolverBenchmark
using NLPModels
using DataFrames
using CSV

# solvers options
atol = 1e-4
rtol = 1e-4
max_eval = -1
max_iter = typemax(Int)
max_time = Dict(100 => 60., 1000 => 60., 2000 => 120., 5000 => 120.)

# parameters 



# dimensions of problem sets
#dimensions = [100, 1000, 2000, 5000]
dimensions = [100, 1000]
NM = [1,2,5,10]
############ run solvers on problem sets ##############
stats = Dict{Int,Any}()
for dim in dimensions
  stats[dim] = Dict{Int,Any}()
  for nm in NM
    ad_problems = (eval(Meta.parse(problem))(n = dim) for problem ∈ OptimizationProblems.meta[!, :name]) # problem set

    #solvers being compared
    solvers = Dict(
      :r2 => model -> R2(model, atol = atol, rtol = rtol, max_eval = max_eval, max_iter = max_iter, max_time = max_time[dim],M=nm),
      :tr => model -> fomo(model, atol = atol, rtol = rtol, max_eval = max_eval, max_iter = max_iter, max_time = max_time[dim],M=nm, β = 0.0,step_backend = tr_step()),
      :fomo_r2 => model -> fomo(model, atol = atol, rtol = rtol, max_eval = max_eval, max_iter = max_iter, max_time = max_time[dim],M=nm, β = 0.9),
      :fomo_tr => model -> fomo(model, atol = atol, rtol = rtol, max_eval = max_eval, max_iter = max_iter, max_time = max_time[dim],M=nm, β = 0.9,step_backend = tr_step()),
    )
    # run solvers over the problem set. Skip problems that are constrained or having dimension not within 20% of dim. 
    stats[dim][nm] =  bmark_solvers(
      solvers, ad_problems,
      skipif=prob -> (!unconstrained(prob) || get_nvar(prob) < dim*0.8 || get_nvar(prob) > dim*1.2 ),
    )
    # add column for average sat\beta to :r2 for consistency
    stats[dim][nm][:r2][!,:avgsatβ] .= 0.
    stats[dim][nm][:tr][!,:avgsatβ] .= 0.
  end
end

export_folder = "docs/src/bench_data/" # change this for the folder path where you want to export the data
for dim in dimensions
  for nm in NM
    for key in keys(stats[dim][nm])
      CSV.write(export_folder*"$(key)_$(dim)_$(nm).csv",stats[dim][nm][key])
    end
  end
end

#####################

using SolverBenchmark
using Plots
using DataFrames
using CSV

###### define paths ######
data_folder = "docs/src/bench_data/"
profile_folder = "docs/src/bench-figures/"
table_folder = "docs/src/fomo-paper-tables/"
ext = ".pdf" # figure format extension

###### load the results #####
dimensions = [100,1000]
NM = [1,2,5,10]
solvers = [:tr,:fomo_tr,:r2,:fomo_r2]
stats = Dict{Int,Any}()
load(solver,dim,nm) = DataFrame(CSV.File(data_folder*solver*"_$(dim)_$(nm).csv"))
for dim in dimensions
  stats[dim] = Dict{Int,Any}()
  for nm in NM
    stats[dim][nm] = Dict([solver => load("$solver",dim,nm) for solver in solvers]...)
  end
end


###### display the results (table) #####
# show solvers execution stats as tables for the problem set of dimension dim_table

# table columns
cols = [:id, :name, :nvar, :objective, :dual_feas, :neval_obj, :neval_grad, :neval_hess, :iter, :elapsed_time, :status, :avgsatβ]
header = Dict(
  :nvar => "n",
  :objective => "f(x)",
  :dual_feas => "‖∇f(x)‖",
  :neval_obj => "# f",
  :neval_grad => "# ∇f",
  :neval_hess => "# ∇²f",
  :elapsed_time => "t",
  :avgsatβ => "avgβmax"
)
dim = 1000 # change this value for any element of dimensions (100,1000,2000,5000)
for solver ∈ solvers
  println("Problems dimension: $dim - Solver:$(solver)")
  pretty_stats(stats[dim][solver][!, cols], hdr_override=header)
end

###### export table with names and avgβmax ########
df_list = []
cols_export = [:name,:avgβmax]
for nm in NM
  push!(df_list,rename!(stats[dim][nm][:fomo_r2][!,cols_export], :avgβmax => "r2_$nm"))
end
for nm in NM
  push!(df_list,rename!(stats[dim][nm][:fomo_tr][!,cols_export], :avgβmax => "tr_$nm"))
end
dfexp = innerjoin(df_list..., on = :name)
pretty_stats(dfexp)
dimmap = [0,100,100,1000,1000,2000,2000,5000,5000]
algmap = [:none,:fomo_r2,:fomo_tr,:fomo_r2,:fomo_tr,:fomo_r2,:fomo_tr,:fomo_r2,:fomo_tr]
failmap(data,i,j) = j!=1 &&  (filter(:name => x -> x == data[i,1],stats[dimmap[j]][algmap[j]])[!,:status][1] != "first_order")
fh = LatexHighlighter(failmap,["cellcolor{black}", "color{white}"])
io = open(table_folder*"avgbetamax.tex")
pretty_stats(io,dfexp,backend = Val(:latex),highlighters = fh)
close(io)

###### export perfomance profiles ######
first_order(df) = df.status .== "first_order"
unbounded(df) = df.status .== "unbounded"
solved(df) = first_order(df) .| unbounded(df)
costnames = ["time"]
costs = [
  #df -> .!solved(df) .* Inf .+ df.elapsed_time,
  df -> .!solved(df) .* Inf .+ df.iter,
  #df -> .!solved(df) .* Inf .+ df.neval_obj .+ df.neval_grad .+ df.neval_hess,
]

gr()
for dim in dimensions
  p_r2 = profile_solvers(Dict(:r2 => stats[dim][:r2], :fomo_r2 => stats[dim][:fomo_r2]), costs, costnames)
  p_tr = profile_solvers(Dict(:r2 => stats[dim][:tr], :fomo_r2 => stats[dim][:fomo_tr]), costs, costnames)
  savefig(p_r2,profile_folder*"R2vsFOMO(R2)_$dim.pdf")
  savefig(p_tr,profile_folder*"TRvsFOMO(TR)_$dim.pdf")
end

p_r2 = profile_solvers(Dict(:r2_1 => stats[dim][1][:r2], :r2_2 => stats[dim][2][:r2],:r2_5 => stats[dim][5][:r2],:r2_10 => stats[dim][10][:r2]), costs, costnames)
p_tr = profile_solvers(Dict(:tr_1 => stats[dim][1][:tr], :tr_2 => stats[dim][2][:tr],:tr_5 => stats[dim][5][:tr],:tr_10 => stats[dim][10][:tr]), costs, costnames)
p_fomo_r2 = profile_solvers(Dict(:fomo_r2_1 => stats[dim][1][:fomo_r2], :fomo_r2_2 => stats[dim][2][:fomo_r2],:fomo_r2_5 => stats[dim][5][:fomo_r2],:fomo_r2_10 => stats[dim][10][:fomo_r2]), costs, costnames)
p_fomo_tr = profile_solvers(Dict(:fomo_tr_1 => stats[dim][1][:fomo_tr], :fomo_tr_2 => stats[dim][2][:fomo_tr],:fomo_tr_5 => stats[dim][5][:fomo_tr],:fomo_tr_10 => stats[dim][10][:fomo_tr]), costs, costnames)
savefig(p_r2,profile_folder*"R2vsFOMO(R2)_$dim.pdf")
savefig(p_tr,profile_folder*"TRvsFOMO(TR)_$dim.pdf")
