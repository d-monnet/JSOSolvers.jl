export adam, AdamSolver

"""
    adam(nlp; kwargs...)

ADAM algorithm for unconstrained optimization.

For advanced usage, first define a `AdamSolver` to preallocate the memory used in the algorithm, and then call `solve!`:

    solver = AdamSolver(nlp)
    solve!(solver, nlp; kwargs...)

# Arguments
- `nlp::AbstractNLPModel{T, V}` is the model to solve, see `NLPModels.jl`.

# Keyword arguments 
- `x::V = nlp.meta.x0`: the initial guess.
- `atol::T = √eps(T)`: absolute tolerance.
- `rtol::T = √eps(T)`: relative tolerance: algorithm stops when ‖∇f(xᵏ)‖ ≤ atol + rtol * ‖∇f(x⁰)‖.
- `η = 0.05`: step size parameters.
- `max_eval::Int = -1`: maximum number of evaluation of the objective function.
- `max_time::Float64 = 30.0`: maximum time limit in seconds.
- `max_iter::Int = typemax(Int)`: maximum number of iterations.
- `β1 = T(0.9) ∈ [0,1)` : constant in the momentum term.
- `β2 = T(0.999) ∈ [0,1)` : constant in the RMSProp term.
- `e = T(1e-8)`: RMSProp epsilon
- `verbose::Int = 0`: if > 0, display iteration details every `verbose` iteration.

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
mutable struct AdamSolver{T, V} <: AbstractOptimizationSolver
  x::V
  ∇f::V
  m::V
  v::V
end

function AdamSolver(nlp::AbstractNLPModel{T, V}) where {T, V}
  x = similar(nlp.meta.x0)
  ∇f = similar(nlp.meta.x0)
  m = fill!(similar(nlp.meta.x0), 0)
  v = fill!(similar(nlp.meta.x0), 0)
  return AdamSolver{T, V}(x, ∇f, m, v)
end

@doc (@doc AdamSolver) function adam(nlp::AbstractNLPModel{T, V}; kwargs...) where {T, V}
  solver = AdamSolver(nlp)
  return solve!(solver, nlp; kwargs...)
end

function SolverCore.reset!(solver::AdamSolver{T}) where {T}
  fill!(solver.m,0)
  fill!(solver.v,0)
  solver
end
SolverCore.reset!(solver::AdamSolver, ::AbstractNLPModel) = reset!(solver)

function SolverCore.solve!(
  solver::AdamSolver{T, V},
  nlp::AbstractNLPModel{T, V},
  stats::GenericExecutionStats{T, V};
  callback = (args...) -> nothing,
  x::V = nlp.meta.x0,
  atol::T = √eps(T),
  rtol::T = √eps(T),
  η = T(0.5),
  e = T(1e-8),
  max_time::Float64 = 30.0,
  max_eval::Int = -1,
  max_iter::Int = typemax(Int),
  β1::T = T(0.9),
  β2::T = T(0.999),
  verbose::Int = 0,
) where {T, V}
  unconstrained(nlp) || error("adam should only be called on unconstrained problems.")

  reset!(stats)
  start_time = time()
  set_time!(stats, 0.0)

  x = solver.x .= x
  ∇fk = solver.∇f
  m = solver.m
  v = solver.v
  set_iter!(stats, 0)
  set_objective!(stats, obj(nlp, x))

  grad!(nlp, x, ∇fk)
  norm_∇fk = norm(∇fk)
  set_dual_residual!(stats, norm_∇fk)
  
  # Stopping criterion: 
  ϵ = atol + rtol * norm_∇fk
  optimal = norm_∇fk ≤ ϵ
  if optimal
    @info("Optimal point found at initial point")
    @info @sprintf "%5s  %7s" "iter" "‖∇f‖" 
    @info @sprintf "%5d   %7.1e" stats.iter norm_∇fk
  end
  if verbose > 0 && mod(stats.iter, verbose) == 0
    @info @sprintf "%5s   %7s" "iter" "‖∇f‖"
    infoline = @sprintf "%5d  %7.1e" stats.iter norm_∇fk 
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
  while !done
    m .= ∇fk .* (T(1) - β1) .+ m .* β1
    v .= ∇fk.^2 .* (T(1) - β2) .+ v .*β2

    x .= x .- η .* (m ./ (1-β1^(stats.iter+1))) ./ ( sqrt.(v ./(1-β2^(stats.iter+1))) .+ e )
    callback(nlp, solver, stats)
    grad!(nlp, x, ∇fk)
    norm_∇fk = norm(∇fk)
    set_iter!(stats, stats.iter + 1)
    set_time!(stats, time() - start_time)
    set_dual_residual!(stats, norm_∇fk)

    if verbose > 0 && mod(stats.iter, verbose) == 0
      @info infoline
      infoline = @sprintf "%5d  %7.1e" stats.iter norm_∇fk
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
    done = stats.status != :unknown
  end

  set_solution!(stats, x)
  return stats
end