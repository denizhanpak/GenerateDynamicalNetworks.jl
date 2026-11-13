# One frame-based simulator for every backend: Euler–Maruyama at a fixed step, nothing saved, the
# state read by the model's decoder once per frame of the specification's clock.

"""The visits one run records: the state entered at each change and when, and whether it failed."""
mutable struct Frames
    state::Vector{Int}
    t::Vector{Float64}
    failed::Bool
end
Frames() = Frames(Int[], Float64[], false)

"""
    frame_steps(m, sample, dtscale) -> (every, dt)

Steps per frame and the step itself: the largest step no coarser than `dtscale / speed(m)` that
lands on every frame.
"""
function frame_steps(m::AbstractNetworkModel, sample, dtscale)
    every = ceil(Int, sample * speed(m) / dtscale)
    return every, sample / every
end

function read_frame!(ig, m::AbstractNetworkModel)
    q = ig.p
    p = project(m, ig.u)
    p === nothing || (ig.u = p)
    ig.iter % q.every == 0 || return nothing
    u = ig.u
    if !in_bounds(m, u)
        q.V.failed = true
        terminate!(ig)
        return nothing
    end
    k = frame_state(m, q, u)
    if k > 0 && (isempty(q.V.state) || last(q.V.state) != k)
        push!(q.V.state, k)
        push!(q.V.t, ig.t)
    end
    return nothing
end

quiet(dt, callback) = (; dt, adaptive=false, save_everystep=false, save_start=false, save_end=false, dense=false,
                         callback, maxiters=typemax(Int))

"""
    simulate(m, spec; T, seeds=1:16, dtscale=0.01)
    simulate(m; T, seeds, sample, dtscale=0.01)

Independent runs of model `m` for time `T`, one per seed, in parallel over threads. Each returns
`(; state, t, failed)`: the states entered, read once per frame of length `sample` (the
specification's), the times they were entered, and whether the run left the model's bounds. The
step is the largest no coarser than `dtscale / speed(m)` that lands on every frame. A run depends
only on its seed, not on the number of threads.
"""
function simulate(m::AbstractNetworkModel; T::Real, seeds, sample::Real, dtscale::Real=0.01)
    every, dt = frame_steps(m, sample, dtscale)
    q = merge(parameters(m), (; every, V=Frames()))
    prob = SDEProblem(drift_function(m), noise_function(m), initial_state(m), (0.0, float(T)), q)
    ens = EnsembleProblem(prob;
        prob_func=(pr, ctx) -> remake(pr; seed=UInt64(seeds[ctx.sim_id]), p=(; pr.p..., V=Frames())),
        output_func=(sol, ctx) -> ((state=sol.prob.p.V.state, t=sol.prob.p.V.t, failed=sol.prob.p.V.failed), false))
    callback = DiscreteCallback((u, t, ig) -> true, ig -> read_frame!(ig, m); save_positions=(false, false))
    return solve(ens, EM(), EnsembleThreads(); trajectories=length(seeds), quiet(dt, callback)...).u
end

simulate(m::AbstractNetworkModel, spec::TransitionSpec; T::Real, seeds=1:16, dtscale::Real=0.01) =
    simulate(m; T, seeds, sample=spec.sample, dtscale)
