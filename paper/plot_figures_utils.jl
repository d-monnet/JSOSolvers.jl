using TikzPictures

function export_performance_profile_tikz(
  x_mat::Matrix{Float64},
  y_mat::Matrix{Float64},
  filename::String;
  file_type = TikzPictures.TIKZ,
  solvernames::Vector{String} = String[],
  xlim::AbstractFloat = 10.0,
  ylim::AbstractFloat = 10.0,
  nxgrad::Int = 5,
  nygrad::Int = 5,
  grid::Bool = true,
  # markers::Vector{S} = String[],
  colours::Vector{String} = String[],
  linestyles::Vector{String} = String[],
  linewidth::AbstractFloat = 1.0,
  xlabel::String = "",
  ylabel::String = "",
  axis_tick_length::AbstractFloat = 0.2,
  lgd_on::Bool = true,
  lgd_pos::Vector = [xlim + 0.5, ylim],
  lgd_plot_length::AbstractFloat = 0.7,
  lgd_v_offset::AbstractFloat = 0.7,
  lgd_plot_offset::AbstractFloat = 0.1,
  lgd_box_length::AbstractFloat = 3.0,
  x_label_offset::AbstractFloat = 1.0,
  y_label_offset::AbstractFloat = 1.0,
  label_val::Vector = [0.2, 0.25, 0.5, 1],
  logscale::Bool = true,
  options::String = "",
  kwargs...,
)
  xlabel_def, ylabel_def, solvernames =
    BenchmarkProfiles.performance_profile_axis_labels(solvernames, size(x_mat, 2), logscale; kwargs...)
  isempty(xlabel) && (xlabel = xlabel_def)
  isempty(ylabel) && (ylabel = ylabel_def)

  y_grad = collect(0.0:(1.0 / (nygrad - 1)):1.0)

  isempty(colours) && (colours = ["black" for _ = 1:size(x_mat, 2)])
  isempty(linestyles) && (linestyles = ["solid" for _ = 1:size(x_mat, 2)])

  #x_mat, y_mat = BenchmarkProfiles.performance_profile_data_mat(T; logscale = logscale, kwargs...)

  # get nice looking graduation on x axis
  xmax, _ = findmax(x_mat[.!isnan.(x_mat)])
  dist = xmax / (nxgrad - 1)
  n = log.(10, dist ./ label_val)
  _, ind = findmin(abs.(n .- round.(n)))
  xgrad_dist = label_val[ind] * 10^round(n[ind])
  x_grad = [0.0, [xgrad_dist * i for i = 1:(nxgrad - 1)]...]
  xmax = max(x_grad[end], xmax)

  # get nice looking graduation on y axis
  dist = 1.0 / (nygrad - 1)
  n = log.(10, dist ./ label_val)
  _, ind = findmin(abs.(n .- round.(n)))
  ygrad_dist = label_val[ind] * 10^round(n[ind])
  y_grad = [0.0, [ygrad_dist * i for i = 1:(nygrad - 1)]...]
  ymax = max(y_grad[end], 1.0)

  to_int(x) = isinteger(x) ? Int(x) : x

  xratio = xlim / xmax
  yratio = ylim / ymax
  io = IOBuffer()

  # axes
  println(io, "\\draw[line width=$linewidth] (0,0) -- ($xlim,0);")
  println(io, "\\node at ($(xlim/2), -$x_label_offset) {$xlabel};")
  println(io, "\\draw[line width=$linewidth] (0,0) -- (0,$ylim);")
  println(io, "\\node at (-$y_label_offset,$(ylim/2)) [rotate = 90]  {$ylabel};")
  # grid
  if grid
    for i in eachindex(x_grad)
      println(io, "\\draw[gray] ($(x_grad[i]*xratio),0) -- ($(x_grad[i]*xratio),$ylim);")
    end
    for i in eachindex(y_grad)
      println(
        io,
        "\\draw[gray] (0,$(y_grad[i]*yratio)) -- ($xlim,$(y_grad[i]*yratio));",
      )
    end
  end
  # axes graduations and labels,
  if logscale
    for i in eachindex(x_grad)
      println(
        io,
        "\\draw[line width=$linewidth] ($(x_grad[i]*xratio),0) -- ($(x_grad[i]*xratio),$axis_tick_length) node [pos=0, below] {\$2^{$(to_int(x_grad[i]))}\$};",
      )
    end
  else
    for i in eachindex(x_grad)
      println(
        io,
        "\\draw[line width=$linewidth] ($(x_grad[i]*xratio),0) -- ($(x_grad[i]*xratio),$axis_tick_length) node [pos=0, below] {$(to_int(x_grad[i]))};",
      )
    end
  end
  for i in eachindex(y_grad)
    println(
      io,
      "\\draw[line width=$linewidth] (0,$(y_grad[i]*yratio)) -- ($axis_tick_length,$(y_grad[i]*yratio)) node [pos=0, left] {$(to_int(y_grad[i]))};",
    )
  end

  # profiles
  for j in eachindex(solvernames)
    drawcmd = "\\draw[line width=$linewidth, $(colours[j]), $(linestyles[j]), line width = $linewidth] "
    drawcmd *= "($(x_mat[1,j]*xratio),$(y_mat[1,j]*yratio))"
    for k = 2:size(x_mat, 1)
      if isnan(x_mat[k, j])
        break
      end
      if y_mat[k, j] > 1 # for some reasons last point of profile is set with y=1.1 by data function...
        drawcmd *= " -- ($(xmax*xratio),$(y_mat[k-1,j]*yratio)) -- ($(xmax*xratio),$(y_mat[k-1,j]*yratio))"
      else
        # if !isempty(markers)
        #   drawcmd *= " -- ($(x_mat[k,j]*xratio),$(y_mat[k-1,j]*yratio)) node[$(colours[j]),draw,$(markers[j]),solid] {} -- ($(x_mat[k,j]*xratio),$(y_mat[k,j]*yratio))"
        # else
        drawcmd *= " -- ($(x_mat[k,j]*xratio),$(y_mat[k-1,j]*yratio)) -- ($(x_mat[k,j]*xratio),$(y_mat[k,j]*yratio))"
        # end
      end
    end
    drawcmd *= ";"
    println(io, drawcmd)
  end

  # legend box
  if lgd_on
    println(
      io,
      "\\draw[line width=$linewidth,fill=white] ($(lgd_pos[1]),$(lgd_pos[2])) rectangle ($(lgd_pos[1]+lgd_box_length),$(lgd_pos[2]-lgd_v_offset*(length(solvernames)+1)));",
    )
    # legend
    for j in eachindex(solvernames)
      legcmd = "\\draw[$(colours[j]), $(linestyles[j]), line width = $linewidth] "
      legcmd *= "($(lgd_pos[1]+lgd_plot_offset),$(lgd_pos[2]-j*lgd_v_offset)) -- ($(lgd_pos[1]+lgd_plot_offset+lgd_plot_length),$(lgd_pos[2]-j*lgd_v_offset)) node [black,pos=1,right] {$(String(solvernames[j]))}"
      # if !isempty(markers)
      #   legcmd *= " node [midway,draw,$(markers[j]),solid] {}"
      # end
      legcmd *= ";"

      println(io, legcmd)
    end
  end

  raw_code = String(take!(io))
  tp = TikzPictures.TikzPicture(raw_code,options=options)
  TikzPictures.save(file_type(filename), tp)
end