include("plot_figures_utils.jl")

function load_fomo_data(
  data_folder::String,
  dimensions::Vector{Int},
  NM::Vector{Int},
  solvers::Vector{Symbol},
)
  stats = Dict{Int, Any}()
  load(solver, dim, nm) = DataFrame(CSV.File(data_folder * solver * "_$(dim)_$(nm).csv"))
  for dim in dimensions
    stats[dim] = Dict{Int, Any}()
    for nm in NM
      stats[dim][nm] = Dict([solver => load("$solver", dim, nm) for solver in solvers]...)
    end
  end
  return stats
end

function export_avg_beta_table(
  stats::Dict{Int, Any},
  solvers::Vector{Symbol},
  dimensions::Vector{Int},
  export_folder::String;
  M = 1,
)
  df_list = []
  cols_export = [:name, :avgβmax]
  for dim in dimensions
    for solver in solvers
      push!(
        df_list,
        rename!(stats[dim][M][solver][!, cols_export], :avgβmax => "$(solver)_$(dim)_$M"),
      )
    end
  end
  dfexp = innerjoin(df_list..., on = :name)
  pretty_stats(dfexp)
  dimmap = [0, vcat([ones(length(solvers)) .* d for d in dimensions]...)...]
  algmap = [:none, vcat([vcat([s for s in solvers]...) for eachindex in dimensions]...)...]
  failmap(data, i, j) =
    j != 1 && (
      filter(:name => x -> x == data[i, 1], stats[dimmap[j]][M][algmap[j]])[!, :status][1] !=
      "first_order"
    )
  fh = LatexHighlighter(failmap, ["cellcolor{black}", "color{white}"])
  io = open(export_folder * "avgbetamax.tex", "w")
  pretty_stats(io, dfexp, backend = Val(:latex), highlighters = fh)
  close(io)
end

function export_performance_profile(
  stats::Dict{Int, Any},
  solvers::Vector{Symbol},
  dimensions::Vector{Int},
  NM,
  cost,
  export_folder::String,
  solvernames;
  file_name = "",
  lgd = Dict([(dim,true) for dim in dimensions]),
  kwargs...
)
  for dim ∈ dimensions
    #### run this for comparing all solvers together
    PALL =
      hcat(cost.(reduce(vcat, [[stats[dim][nm][solver] for solver in solvers] for nm in NM]))...)
    x_mat, y_mat = BenchmarkProfiles.performance_profile_data_mat(PALL)

    #names = reduce(vcat, [["$(solver)_$nm" for solver in solvers] for nm in NM])
    names = length(NM) == 1 ? reduce(vcat, [[solvernames[solver] for solver in solvers] for nm in NM]) : reduce(vcat, [[solvernames[solver]*"\$M = $nm\$" for solver in solvers] for nm in NM])
    if file_name == ""
      file_name = names[1]
      for i in 2:length(names)
       file_name *= "vs"*names[i]
      end  
    end
    export_performance_profile_tikz(
      x_mat,
      y_mat,
      export_folder * file_name * "_$(dim)";
      file_type = PDF,
      solvernames = names,
      options = "scale=0.7",
      y_label_offset = 1.5,
      lgd_on = lgd[dim],
      kwargs...
    )
  end
end

function set_status_nomin(
  stats::Dict{Int, Any},
  solve_tol,
)
  for dim in keys(stats)
    obj = reduce(vcat, [[stats[dim][M][solver][!, :objective] for solver in keys(stats[dim][M])] for M in keys(stats[dim])])
    min_obj = minimum(stack(obj), dims = 2)[:, 1]
    for M in keys(stats[dim])
      for solver in keys(stats[dim][M])
        s = stats[dim][M][solver]
          for r = 1:nrow(s)
          obj_single = s[r, :objective]
          if obj_single > min_obj[r] * (1 + sign(min_obj[r]) * solve_tol)
            s[r, :status] = "no_min"
          end
        end
      end
    end
  end
end


function export_histogram_data(
  stats::Dict{Int, Any},
  solvers::Vector{Symbol},
  dimensions::Vector{Int},
  bins,
  export_folder::String,
)
  hist_colnames = ["solver", "dim", "M", ["bin_$i" for i = 1:(length(bins) - 1)]...]
  hist_types = [Symbol, [Int for _ = 1:(length(hist_colnames) - 1)]...]
  hist = DataFrame([T[] for T in hist_types], hist_colnames)
  avg_colnames = ["solver", "dim", "M", "avg_Beta"]
  avg_types = [Symbol, Int, Int, Float64]
  avgbetamax = DataFrame([T[] for T in avg_types], avg_colnames)
  for solver in solvers
    for dim in dimensions
      for nm in NM
        hist_val = [
          size(
            filter(x -> x.avgβmax > bins[i] && x.avgβmax ≤ bins[i + 1], stats[dim][nm][solver]),
            1,
          ) for i = 1:(length(bins) - 1)
        ]
        push!(hist, [solver, dim, nm, hist_val...])
        avg_beta = mean(filter(x -> !isnan(x.avgβmax), stats[dim][nm][solver])[!, :avgβmax])
        push!(avgbetamax, [solver, dim, nm, avg_beta])
      end
    end
  end
  pretty_stats(hist)
  pretty_stats(avgbetamax)
  CSV.write(export_folder * "histograms_data.csv", hist)
  CSV.write(export_folder * "avgbetamax_data.csv", avgbetamax)
end