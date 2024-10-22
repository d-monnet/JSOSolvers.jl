function run_benchmark(dimensions::Vector{Int},
  NM::Vector{Int},
  solver_symbs::Vector{Symbol},
  β,
  atol,
  rtol,
  max_eval,
  max_iter,
  max_time::Dict)

  stats = Dict{Int,Any}()
  for dim in dimensions
    stats[dim] = Dict{Int,Any}()
    for nm in NM
      ad_problems = (eval(Meta.parse(problem))(n = dim) for problem ∈ OptimizationProblems.meta[!, :name]) # problem set

      #solvers being compared
      βval(solver) = solver in [:tr, :r2] ? 0.0 : β  
      step_backend(solver) = solver in [:tr, :fomo_tr] ? tr_step() : r2_step()
      solvers = Dict(
        solver => model -> fomo(model, atol = atol, rtol = rtol, max_eval = max_eval, max_iter = max_iter, max_time = max_time[dim], M=nm, β = βval(solver), step_backend = step_backend(solver)) for solver in solver_symbs
        #:fomo_tr => model -> fomo(model, atol = atol, rtol = rtol, max_eval = max_eval, max_iter = max_iter, max_time = max_time[dim],M=nm, β = 0.9,step_backend = tr_step())
      )
      # run solvers over the problem set. Skip problems that are constrained or having dimension not within 20% of specified dimension. 
      stats[dim][nm] =  bmark_solvers(
        solvers, ad_problems,
        skipif=prob -> (!unconstrained(prob) || get_nvar(prob) < dim*0.8 || get_nvar(prob) > dim*1.2 ),
      )
      # add column for average sat\beta to :r2 and :tr for consistency
      for s in solver_symbs 
        if s == :r2 || s == :tr
          stats[dim][nm][s][!,:avgβmax] .= 0.
        end
      end
    
    end
  end
  return stats
end

function save_data(path::String, stats::Dict)

  for dim in keys(stats)
    for nm in keys(stats[dim])
      for key in keys(stats[dim][nm])
        CSV.write(path*"$(key)_$(dim)_$(nm).csv",stats[dim][nm][key])
      end
    end
  end
end