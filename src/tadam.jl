export tadam, TadamSolver

"""
    tadam(nlp; kwargs...)

Trust-region embeded ADAM (TADAM) algorithm for unconstrained optimization.

For advanced usage, first define a `TadamSolver` to preallocate the memory used in the algorithm, and then call `solve!`:

    solver = TadamSolver(nlp)
    solve!(solver, nlp; kwargs...)

# Arguments
- `nlp::AbstractNLPModel{T, V}` is the model to solve, see `NLPModels.jl`.

# Keyword arguments 
- `x::V = nlp.meta.x0`: the initial guess.
- `atol::T = √eps(T)`: absolute tolerance.
- `rtol::T = √eps(T)`: relative tolerance: algorithm stops when ‖∇f(xᵏ)‖ ≤ atol + rtol * ‖∇f(x⁰)‖.
- `η1 = eps(T)^(1/4)`, `η2 = T(0.2)`: step acceptance parameters.
- `γ1 = T(0.5)`, `γ2 = T(1.1)`: regularization update parameters.
- `Δmax = 1/eps(T)`: step parameter for tadam algorithm.
- `max_eval::Int = -1`: maximum number of evaluation of the objective function.
- `max_time::Float64 = 30.0`: maximum time limit in seconds.
- `max_iter::Int = typemax(Int)`: maximum number of iterations.
- `β1 = T(0.9) ∈ [0,1)` : constant in the momentum term.
- `β2 = T(0.999) ∈ [0,1)` : constant in the RMSProp term.
- `e = T(1e-8)` : RMSProp epsilon
- `verbose::Int = 0`: if > 0, display iteration details every `verbose` iteration.
- `backend = qr()`: model-based method employed. Options are `qr()` for quadratic regulation and `tr()` for trust-region

# Output
The value returned is a `GenericExecutionStats`, see `SolverCore.jl`.

# Callback
The callback is called at each iteration.
The expected signature of the callback is `callback(nlp, solver, stats)`, and its output is ignored.
Changing any of the input arguments will affect the subsequent iterations.
In particular, setting `stats.status = :user` will stop the algorithm.
All relevant information should be available in `nlp` and `solver`.
Notably, you can access, and modify, the following:
- `solver.x`: current iterate;
- `solver.gx`: current gradient;
- `stats`: structure holding the output of the algorithm (`GenericExecutionStats`), which contains, among other things:
  - `stats.dual_feas`: norm of current gradient;
  - `stats.iter`: current iteration counter;
  - `stats.objective`: current objective function value;
  - `stats.status`: current status of the algorithm. Should be `:unknown` unless the algorithm has attained a stopping criterion. Changing this to anything will stop the algorithm, but you should use `:user` to properly indicate the intention.
  - `stats.elapsed_time`: elapsed time in seconds.

# Examples
```jldoctest
using JSOSolvers, ADNLPModels
nlp = ADNLPModel(x -> sum(x.^2), ones(3))
stats = fomo(nlp)

# output

"Execution stats: first-order stationary"
```

```jldoctest
using JSOSolvers, ADNLPModels
nlp = ADNLPModel(x -> sum(x.^2), ones(3))
solver = FomoSolver(nlp);
stats = solve!(solver, nlp)

# output

"Execution stats: first-order stationary"
```
"""
mutable struct TadamSolver{T, V} <: AbstractOptimizationSolver
  x::V
  ∇f::V
  c::V
  m::V
  v::V
  s::V
end

function TadamSolver(nlp::AbstractNLPModel{T, V}) where {T, V}
  x = similar(nlp.meta.x0)
  ∇f = similar(nlp.meta.x0)
  c = similar(nlp.meta.x0)
  m = fill!(similar(nlp.meta.x0), 0)
  v = fill!(similar(nlp.meta.x0), 0)
  s = fill!(similar(nlp.meta.x0), 0)
  return TadamSolver{T, V}(x, ∇f, c, m, v, s)
  return TadamSolver{T, V}(x, ∇f, c, m, v, s)
end

@doc (@doc TadamSolver) function tadam(nlp::AbstractNLPModel{T, V}; kwargs...) where {T, V}
  solver = TadamSolver(nlp)
  return solve!(solver, nlp; kwargs...)
end

function SolverCore.reset!(solver::TadamSolver{T}) where {T}
  fill!(solver.m,0)
  fill!(solver.v,0)
  solver
end
SolverCore.reset!(solver::TadamSolver, ::AbstractNLPModel) = reset!(solver)

function SolverCore.solve!(
  solver::TadamSolver{T, V},
  nlp::AbstractNLPModel{T, V},
  stats::GenericExecutionStats{T, V};
  callback = (args...) -> nothing,
  x::V = nlp.meta.x0,
  atol::T = √eps(T),
  rtol::T = √eps(T),
  η1 = eps(T)^(1 / 4),
  η2 = T(0.95),
  η2 = T(0.95),
  γ1 = T(0.5),
  γ2 = T(2),
  Δmax = 1/eps(T),
  Δmax = 1/eps(T),
  max_time::Float64 = 30.0,
  max_eval::Int = -1,
  max_iter::Int = typemax(Int),
  β1::T = T(0.9),
  β2::T = T(0.999),
  e::T = T(1e-8),
  verbose::Int = 0,
) where {T, V}
  unconstrained(nlp) || error("tadam should only be called on unconstrained problems.")

  reset!(stats)
  start_time = time()
  set_time!(stats, 0.0)

  x = solver.x .= x
  ∇fk = solver.∇f
  c = solver.c
  m = solver.m
  v = solver.v
  s = solver.s
  set_iter!(stats, 0)
  set_objective!(stats, obj(nlp, x))

  grad!(nlp, x, ∇fk)
  norm_∇fk = norm(∇fk)
  set_dual_residual!(stats, norm_∇fk)

  Δk = norm_∇fk/2^round(log2(norm_∇fk + 1))
  
  # Stopping criterion: 
  ϵ = atol + rtol * norm_∇fk
  optimal = norm_∇fk ≤ ϵ
  if optimal
    @info("Optimal point found at initial point")
    @info @sprintf "%5s  %9s  %7s  %7s " "iter" "f" "‖∇f‖" "Δ"
    @info @sprintf "%5d  %9.2e  %7.1e  %7.1e" stats.iter stats.objective norm_∇fk Δk
    @info @sprintf "%5s  %9s  %7s  %7s " "iter" "f" "‖∇f‖" "Δ"
    @info @sprintf "%5d  %9.2e  %7.1e  %7.1e" stats.iter stats.objective norm_∇fk Δk
  end
  if verbose > 0 && mod(stats.iter, verbose) == 0
    @info @sprintf "%5s  %9s  %7s  %7s  %7s" "iter" "f" "‖∇f‖" "α" "satβ1"
    infoline = @sprintf "%5d  %9.2e  %7.1e  %7.1e  %7.1e" stats.iter stats.objective norm_∇fk Δk NaN
  end

  set_status!(
    stats,
    get_status(
      nlp,
      elapsed_time = stats.elapsed_time,
      optimal = optimal,
      max_eval = max_eval,
      iter = stats.iter,
      max_iter = max_iter,
      max_time = max_time,
    ),
  )

  callback(nlp, solver, stats)

  done = stats.status != :unknown
  
  satβ1 = T(0)
  siter=1 # nb of successful iteration
  μ = T(0)
  while !done
    solve_tadam_subproblem!(s, ∇fk, m, v, Δk, satβ1, β1, β2, e, siter)
    c .= x .+ s

    ΔTk = dot(-∇fk, s) - T(0.5)*dot(s.^2, sqrt.(v) .+ 1e-8)
    fck = obj(nlp, c)
    if fck == -Inf
      set_status!(stats, :unbounded)
      break
    end

    ρk = (stats.objective - fck) / ΔTk
    if ρk >= η2
      Δk = min(Δmax, γ2 * Δk)
    elseif ρk < η1
      Δk = Δk * γ1
    end
    if Δk == 0.
      set_status!(stats, :exception)
    end
    # Acceptance of the new candidate
    if ρk >= η1
      siter += 1
      x .= c
      μ = Δk * (T(1) - β1) + Δk * β1
      m .= (Δk/μ) .* ∇fk .* (T(1) - β1) .+ m .* β1
      #m .= ∇fk .* (T(1) - β1) .+ m .* β1
      v .= ∇fk.^2 .* (T(1) - β2) .+ v .*β2
      cout = callback(nlp, solver, stats)
      if !isnothing(cout)
        set_objective!(stats, obj(nlp,x))
      else
        set_objective!(stats, fck)
      end
      grad!(nlp, x, ∇fk)
      dotprod = dot(∇fk,m)
      satβ1 = find_beta(β1, dotprod, norm_∇fk)
      norm_∇fk = norm(∇fk)
      Δk = μ
    end

    set_iter!(stats, stats.iter + 1)
    set_time!(stats, time() - start_time)
    set_dual_residual!(stats, norm_∇fk)
    optimal = norm_∇fk ≤ ϵ

    if verbose > 0 && mod(stats.iter, verbose) == 0
      @info infoline
      infoline = @sprintf "%5d  %9.2e  %7.1e  %7.1e  %7.1e" stats.iter stats.objective norm_∇fk Δk satβ1
      infoline = @sprintf "%5d  %9.2e  %7.1e  %7.1e  %7.1e" stats.iter stats.objective norm_∇fk Δk satβ1
    end

    set_status!(
      stats,
      get_status(
        nlp,
        elapsed_time = stats.elapsed_time,
        optimal = optimal,
        max_eval = max_eval,
        iter = stats.iter,
        max_iter = max_iter,
        max_time = max_time,
      ),
    )

    #callback(nlp, solver, stats)

    done = stats.status != :unknown
  end

  set_solution!(stats, x)
  return stats
end

"""
  solve_tadam_subproblem!(s, ∇fk, m, v, Δk, satβ1)
Compute 
argmin d^Ts + s^T diag(sqrt.(v)) 
s.t.   ||s||∞ <= Δk      
with d = (1-satβ1) * ∇fk + satβ1 * m  
Stores the argmin in `s`.
"""
function solve_tadam_subproblem!(s::V, ∇fk::V, m::V, v::V, Δk::T, satβ1::T, β1::T, β2::T, e::T, siter::Int) where {V, T}
  s .= min.(Δk , max.(-Δk , -(((1-satβ1) .* ∇fk .+ satβ1 .* m) ./ (1-β1^siter) ) ./ ( sqrt.(v ./ (1 - β2^siter)) .+ e ) ) )
end