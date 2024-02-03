using TikzPictures
using BenchmarkProfiles
using CSV
using DataFrames

include("docs/src/fomo-paper-figures/plot_utils.jl")

data_path = "docs/src/fomo-paper-data/"
dimensions = [100,1000,2000,5000]
lgd = Dict(100 => true, 1000 => false,2000 => false,5000 => false)
lgd_pos = [6.,4.]
lgd_box_length = 4.

for dim ∈ dimensions
  fomo_tr = DataFrame(CSV.File(data_path*"fomo_tr_$(dim).csv"))
  TR = DataFrame(CSV.File(data_path*"tr_$(dim).csv"))
  fomo_r2 = DataFrame(CSV.File(data_path*"fomo_r2_$(dim).csv"))
  r2 = DataFrame(CSV.File(data_path*"r2_$(dim).csv"))

  first_order(df) = df.status .== "first_order"
  unbounded(df) = df.status .== "unbounded"
  solved(df) = first_order(df) .| unbounded(df)
  cost(df) = .!solved(df) .* Inf .+ df.elapsed_time

  PTR = hcat([cost(fomo_tr),cost(TR)]...)
  PR2 = hcat([cost(fomo_r2),cost(r2)]...)

  x_mat_tr, y_mat_tr = BenchmarkProfiles.performance_profile_data_mat(PTR)
  x_mat_r2, y_mat_r2 = BenchmarkProfiles.performance_profile_data_mat(PR2)

  if size(x_mat_tr,1) > size(x_mat_r2,1)
    x_mat_r2 = vcat([x_mat_r2,NaN .* ones(size(x_mat_tr,1) - size(x_mat_r2,1),2)]...)
    y_mat_r2 = vcat([y_mat_r2,NaN .* ones(size(y_mat_tr,1) - size(y_mat_r2,1),2)]...)
  end
  if size(x_mat_tr,1) < size(x_mat_r2,1)
    x_mat_tr = vcat([x_mat_tr,NaN .* ones(size(x_mat_r2,1) - size(x_mat_tr,1),2)]...)
    y_mat_tr = vcat([y_mat_tr,NaN .* ones(size(y_mat_r2,1) - size(y_mat_tr,1),2)]...)
  end

  x_mat = hcat([x_mat_tr,x_mat_r2]...)
  y_mat = hcat([y_mat_tr,y_mat_r2]...)

  colours = ["blue","blue","red","red"]
  styles = ["dashed","solid","dashed","solid"]
  names = ["FOMO(TR)","TR","FOMO(R2)","R2"]

  export_performance_profile_tikz(x_mat,y_mat,"docs/src/fomo-paper-figures/Time_perf_dim_$(dim)";
  file_type=PDF,colours = colours,linestyles = styles,solvernames=names,options="scale=0.7",
  y_label_offset=1.5,lgd_on = lgd[dim], lgd_pos=lgd_pos, lgd_box_length = lgd_box_length )
end