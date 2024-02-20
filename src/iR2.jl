export iR2, iR2Solver

#DM: what does iR2 stands for ?
"""
    iR2(nlp; kwargs...)

A stochastic first-order quadratic regularization method for unconstrained optimization.

For advanced usage, first define a `iR2Solver` to preallocate the memory used in the algorithm, and then call `solve!`:

    solver = iR2Solver(nlp)
    solve!(solver, nlp; kwargs...)

# Arguments
- `nlp::AbstractNLPModel{T, V}` is the model to solve, see `NLPModels.jl`.

# Keyword arguments 
- `x::V = nlp.meta.x0`: the initial guess.
- `atol::T = √eps(T)`: absolute tolerance.
- `rtol::T = √eps(T)`: relative tolerance: algorithm stops when ‖∇f(xᵏ)‖ ≤ atol + rtol * ‖∇f(x⁰)‖.
- `η1 = eps(T)^(1/4)`, `η2 = T(0.95)`: step acceptance parameters.
- `λ = T(2)`, λ > 1 regularization update parameters. 
- `σmin = eps(T)`: step parameter for iR2 algorithm.
- `max_eval::Int = -1`: maximum number of evaluation of the objective function.
- `max_time::Float64 = 30.0`: maximum time limit in seconds.
- `max_iter::Int = typemax(Int)`: maximum number of iterations.
- `β = T(0) ∈ [0,1]` is the constant in the momentum term. If `β == 0`, iR2 does not use momentum.
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
stats = iR2(nlp)

# output

"Execution stats: first-order stationary"
```

```jldoctest
using JSOSolvers, ADNLPModels
nlp = ADNLPModel(x -> sum(x.^2), ones(3))
solver = iR2Solver(nlp);
stats = solve!(solver, nlp)

# output

"Execution stats: first-order stationary"
```
"""
mutable struct iR2Solver{T, V} <: AbstractOptimizationSolver
  x::V
  gx::V
  cx::V
  d::V   # used for momentum term
  σ::T
end

function iR2Solver(nlp::AbstractNLPModel{T, V}) where {T, V}
  x = similar(nlp.meta.x0)
  gx = similar(nlp.meta.x0)
  cx = similar(nlp.meta.x0)
  d = fill!(similar(nlp.meta.x0), 0)
  σ= zero(T) # init it to zero for now 
  return iR2Solver{T, V}(x, gx, cx, d, σ)
end

@doc (@doc iR2Solver) function iR2(nlp::AbstractNLPModel{T, V}; kwargs...) where {T, V}
  solver = iR2Solver(nlp)
  return solve!(solver, nlp; kwargs...)
end

function SolverCore.reset!(solver::iR2Solver{T}) where {T}
  solver.d .= zero(T)
  solver
end
SolverCore.reset!(solver::iR2Solver, ::AbstractNLPModel) = reset!(solver)

function SolverCore.solve!(
  solver::iR2Solver{T, V},
  nlp::AbstractNLPModel{T, V},
  stats::GenericExecutionStats{T, V};
  callback = (args...) -> nothing,
  x::V = nlp.meta.x0,
  atol::T = √eps(T),
  rtol::T = √eps(T),
  η1 = eps(T)^(1 / 4),
  η2 = T(0.95),
  λ = T(2),
  σmin = zero(T), # μmin = σmin to match the paper
  max_time::Float64 = 30.0,
  max_eval::Int = -1,
  max_iter::Int = typemax(Int),
  β::T = T(0),
  verbose::Int = 0,
) where {T, V}
  unconstrained(nlp) || error("iR2 should only be called on unconstrained problems.")
  
  reset!(stats)
  start_time = time()
  set_time!(stats, 0.0)

  x = solver.x .= x
  ∇fk = solver.gx
  ck = solver.cx
  d = solver.d
  σk = solver.σ 

  set_iter!(stats, 0)
  set_objective!(stats, obj(nlp, x))

  grad!(nlp, x, ∇fk)
  norm_∇fk = norm(∇fk)
  set_dual_residual!(stats, norm_∇fk)

  μk = 2^round(log2(norm_∇fk + 1)) / norm_∇fk #TODO confirm if this is the correct initialization DM: Looks ok to me
  σk=  μk * norm_∇fk

  # Stopping criterion: 
  ϵ = atol + rtol * norm_∇fk
  optimal = norm_∇fk ≤ ϵ
  if optimal
    @info("Optimal point found at initial point")
    @info @sprintf "%5s  %9s  %7s  %7s " "iter" "f" "‖∇f‖" "μ"
    @info @sprintf "%5d  %9.2e  %7.1e  %7.1e" stats.iter stats.objective norm_∇fk μk
  end
  if verbose > 0 && mod(stats.iter, verbose) == 0
    @info @sprintf "%5s  %9s  %7s  %7s " "iter" "f" "‖∇f‖" "μ"
    infoline = @sprintf "%5d  %9.2e  %7.1e  %7.1e" stats.iter stats.objective norm_∇fk μk
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

  solver.σ =  σk 
  callback(nlp, solver, stats)
  σk = solver.σ #TODO do I need this here?

  done = stats.status != :unknown

  while !done
    #TODO unlike R2 since our data is stochastic we need to recompute the gradient and objective and not used the passed 
    set_objective!(stats, obj(nlp, x)) #TODO confirm with Prof.Orban
    grad!(nlp, x, ∇fk)
    norm_∇fk = norm(∇fk)
    σk =  μk * norm_∇fk # TODO Prof. Orban, do we need to update σk here (since the norm is different in for example deep learning models)
    #DM: you can update here, the main thing is having it up to date whenever it is used.
    #TODO prof. Orban, do we need to update the dual residual here?
    # set_dual_residual!(stats, norm_∇fk)
    # optimal = norm_∇fk ≤ ϵ #todo we need to check
    # we will be slower but more accurate  and no need to do them in the callback 
    #DM: I would suggest to update the dual residual with an upper bound on the norm of the gradient that you can derive from the error bound on the gradient (if available).
    
    #TODO rewrite the following to use the momentum term
    #DM: what's wrong with the if block below ?
    if β == 0
      ck .= x .- (∇fk ./ σk)
    else # momentum term
      d .= ∇fk .* (T(1) - β) .+ d .* β
      ck .= x .- (d ./ σk)
    end

    ΔTk = norm_∇fk * μk #TODO OR  ΔTk = norm_∇fk^2 / σk  ?  Prof. Orban
    #DM: you will have the answer once you are sure about the convergence analysis you wrote in the draft.
    fck = obj(nlp, ck)

    if fck == -Inf
      set_status!(stats, :unbounded)
      break
    end

    ρk = (stats.objective - fck) / ΔTk
    #DM:ρk is consistent with your algorithm (as in the report), however you might want to consider ρk = (obj(nlp,xk) - fck) / ΔTk in mini-batch context to stabilize the algorithm. 

    # Update regularization parameters and Acceptance of the new candidate
    if ρk >= η1 && σk >= η2  # TODO if we move the μ^-1 to the left side DM: I do not undertand
      μk = max(σmin,  μk / λ )
      x .= ck
      set_objective!(stats, fck)
      grad!(nlp, x, ∇fk)
      norm_∇fk = norm(∇fk)
    else
      μk = μk * λ
    end

    set_iter!(stats, stats.iter + 1)
    set_time!(stats, time() - start_time)
    set_dual_residual!(stats, norm_∇fk)
    optimal = norm_∇fk ≤ ϵ
    
    σk = μk * norm_∇fk # this is different from R2  #TODO Prof. Orban, do we need to update σk here or at the begining of the loop or both places ?
    #DM: Looks to me it is enough to update it at the beginning of the loop


    
    if verbose > 0 && mod(stats.iter, verbose) == 0
      @info infoline
      infoline = @sprintf "%5d  %9.2e  %7.1e  %7.1e" stats.iter stats.objective norm_∇fk μk
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
    solver.σ= σk
    callback(nlp, solver, stats)
    σk = solver.σ

    done = stats.status != :unknown
  end

  set_solution!(stats, x)
  return stats
end