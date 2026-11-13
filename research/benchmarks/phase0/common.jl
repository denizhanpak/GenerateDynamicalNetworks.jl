# Shared by the Phase 0 benchmarks: the worm ring, one generic Euler-Maruyama kernel, and an online
# version of the Phase 1 measurements. A visit starts when the largest activation passes `edge`
# (`latch` in phase1_core.jl), the first visit is dropped, and the open last one never counts.
using LinearAlgebra, Random, Statistics, StaticArrays, Printf
using SHCTools, Roots

const WORM = (names=["Forward", "Pause 1", "Reverse", "Pause 2"], τ=[5.329, 0.441, 1.945, 0.208])

"""The Phase 1 prototype: one shared α, with ν setting each dwell."""
function ring_model(τ; ε, κ=0.1, νmin=2.0)
    n, β = length(τ), ones(length(τ))
    αᵢ(i, ν) = first(lv_params(τ[i], ε, β[i], ν; κ))
    α = maximum(αᵢ(i, νmin) for i in 1:n)
    ν = [find_zero(v -> αᵢ(i, v) - α, (νmin, 1e6)) for i in 1:n]
    return (; τ, β, κ, ε=fill(float(ε), n), α=fill(α, n), ν, ρ=createcycle(fill(α, n), β, ν))
end

"""SVector parameters of a model, for the kernels."""
function lv_static(m)
    N = length(m.α)
    return (; α=SVector{N}(m.α), ρ=SMatrix{N,N}(m.ρ), ε=SVector{N}(m.ε))
end
lv_drift(x, p) = x .* (p.α - p.ρ * x)
identity_readout(x, p) = x

struct Tally
    sum::Vector{Float64}
    sumsq::Vector{Float64}
    count::Vector{Int}
    moves::Matrix{Int}
end
Tally(n) = Tally(zeros(n), zeros(n), zeros(Int, n), zeros(Int, n, n))

"""Advances `n` steps, returning the tally. `stride` is how often activation is read."""
function em_tally(drift::F, readout::G, x0::SVector{N,T}, p, σ::SVector{N,T}, dt, nsteps; seed,
        edge=0.5, lower=zero(T), stride=1) where {F,G,N,T}
    rng = Xoshiro(seed)
    tally = Tally(N)
    x, step_noise = x0, σ .* sqrt(dt)
    state, entries, entered = 0, 0, 0.0
    @inbounds for k in 1:nsteps
        x = x + dt * drift(x, p) + step_noise .* randn(rng, SVector{N,T})
        x = max.(x, lower)
        k % stride == 0 || continue
        v, i = findmax(readout(x, p))
        if v > edge && i != state
            t = k * dt
            entries += 1
            if entries >= 3    # the first visit is dropped, so dwells start at the second
                d = t - entered
                tally.sum[state] += d
                tally.sumsq[state] += d^2
                tally.count[state] += 1
                tally.moves[state, i] += 1
            end
            state, entered = i, t
        end
    end
    return tally
end

"""Runs `f(seed)` for each seed on `nt` tasks."""
function on_threads(f, seeds, nt)
    results = Vector{Any}(undef, length(seeds))
    chunks = Iterators.partition(eachindex(seeds), max(1, ceil(Int, length(seeds) / nt)))
    @sync for chunk in chunks
        Threads.@spawn for i in chunk
            results[i] = f(seeds[i])
        end
    end
    return [r for r in results]
end

"""Pooled mean dwell and its standard error, over tallies."""
function pooled(tallies)
    n = sum(t.count for t in tallies)
    s = sum(t.sum for t in tallies)
    q = sum(t.sumsq for t in tallies)
    mean_d = s ./ n
    return (; mean=mean_d, sem=sqrt.(max.(q ./ n .- mean_d .^ 2, 0) ./ n), n)
end
