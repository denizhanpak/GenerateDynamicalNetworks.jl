# Two-population backend (Ashwin & Postlethwaite 2016, appendix A): one cell per state and one per edge,
# one global speed, a signed ν per edge (negative: heteroclinic; positive: excitable), one shared noise,
# no hidden states.

# The construction's fixed constants (their A, C, D, E, F), the scale of the signed ν, the contraction the
# seed divides by, and the step-size rule's rate as a multiple of the speed.
const TWOPOP = (A=0.5, C=2.0, D=10.0, E=4.0, F=2.0, ν0=0.01, contract=1.5, step=2.5)

"""
    TwoPopModel{N}

Ashwin & Postlethwaite's two-population network on a graph with `N` edges: state cells `p` and edge
cells `y`, at speed `s`, with `B[k] = 1.5 − ν[k]` for edge `k` (`from[k] → to[k]`). Read by the state
cell of largest |p|. Built by `fit(spec, TwoPopulation())` or `TwoPopModel(spec, θ)`.
"""
struct TwoPopModel{N} <: AbstractNetworkModel
    P::Matrix{Float64}
    visible::Int
    α::Vector{Float64}
    s::Float64
    B::NTuple{N,Float64}
    from::NTuple{N,Int}
    to::NTuple{N,Int}
    ε::Float64
end

function twopop_model(spec, edges, s, ν)
    N = length(edges)
    return TwoPopModel{N}(Matrix{Float64}(spec.P), size(spec.P, 1), [TWOPOP.step * s], s, ntuple(k -> 1.5 - ν[k], N),
                          ntuple(k -> edges[k][1], N), ntuple(k -> edges[k][2], N), float(spec.noise))
end
"""θ = [log s; log(−ν)], the heteroclinic side, for the correction passes."""
twopop_log(spec, edges, θ) = twopop_model(spec, edges, exp(θ[1]), -exp.(θ[2:end]))
"""θ = [log s; asinh(ν/ν₀)], either side of the bifurcation, for the search."""
twopop_signed(spec, edges, θ) = twopop_model(spec, edges, exp(θ[1]), TWOPOP.ν0 .* sinh.(θ[2:end]))

"""
    TwoPopModel(spec, θ)

The model with `θ = [log s; asinh(ν / 0.01)]`, one ν per edge in `edge_list` order.
"""
TwoPopModel(spec::TransitionSpec, θ::AbstractVector) = twopop_signed(spec, edge_list(spec.P), θ)

twopop_drift(z, q, t) = twopop_drift(z, q, q.size)
function twopop_drift(z::SVector{M}, q, ::Val{nv}) where {M,nv}
    ne = M - nv
    p2, y2 = sum(j -> z[j]^2, 1:nv), sum(k -> z[nv+k]^2, 1:ne)
    p4 = p2^2
    dp = ntuple(Val(nv)) do j
        out = sum(k -> q.from[k] == j ? z[nv+k]^2 * z[j] * z[q.to[k]] : 0.0, 1:ne)
        inn = sum(k -> q.to[k] == j ? z[nv+k]^2 * z[q.from[k]]^2 : 0.0, 1:ne)
        q.s * (z[j] * (TWOPOP.F * (1 - p2) + TWOPOP.D * (z[j]^2 * p2 - p4)) + TWOPOP.E * (inn - out))
    end
    dy = ntuple(Val(ne)) do k
        y = z[nv+k]
        λ = TWOPOP.A - q.B[k] * z[q.from[k]]^2 + TWOPOP.C * (y2 - y^2)
        -q.s * y * ((y^2 - 1)^2 + λ)
    end
    return SVector{M}(dp..., dy...)
end

drift_function(::TwoPopModel) = twopop_drift
parameters(m::TwoPopModel) = (; nv=m.visible, size=Val(m.visible), m.s, m.B, m.from, m.to, m.ε)
initial_state(m::TwoPopModel{N}) where {N} = SVector{m.visible + N}(ntuple(i -> i == 1 ? 1.0 : m.ε, m.visible + N))
speed(m::TwoPopModel) = maximum(m.α)
frame_state(::TwoPopModel, q, z) = findmax(i -> abs(z[i]), 1:q.nv)[2]

"""
    TwoPopulation(; passes=4, runs=16, T=nothing, budget=500, popsize=12, seconds=300.0)

The two-population backend's fit: a seed from the Horchler dwell design ([`edge_seed_rates`](@ref)),
`passes` correction passes of `runs` runs of length `T` (default 125Στ) on the heteroclinic side, and
if the best of those fails a check on 32 held-out runs, a CMA-ES search over the signed ν (either side
of the bifurcation) of at most `budget` evaluations or `seconds` seconds, then a short polish if every ν
is still negative. (The prototype stopped at 120 s.) A binding time limit makes the result depend on
the machine's speed; `seconds=Inf` limits the search by `budget` alone.

    fit(spec, TwoPopulation(); verbose=false) -> TwoPopModel

!!! warning "Status: experimental"
    Fails the acceptance protocol on half or more of the benchmark cases; see the Results page of the documentation.
"""
struct TwoPopulation <: AbstractBackend
    passes::Int
    runs::Int
    T::Union{Float64,Nothing}
    budget::Int
    popsize::Int
    seconds::Float64
end
TwoPopulation(; passes::Integer=4, runs::Integer=16, T::Union{Real,Nothing}=nothing, budget::Integer=500, popsize::Integer=12,
              seconds::Real=300.0) = TwoPopulation(passes, runs, T === nothing ? nothing : float(T), budget, popsize, seconds)

function fit(spec::TransitionSpec, b::TwoPopulation; verbose::Bool=false)
    edges = edge_list(spec.P)
    ne = length(edges)
    T = b.T === nothing ? 125sum(spec.τ) : b.T
    r = edge_seed_rates(spec, edges)
    logθ0 = [log(r.α / TWOPOP.contract); r.logs .+ log(TWOPOP.contract)]
    llo, lhi = [log(0.05); fill(log(1e-4), ne)], [log(200.0); fill(log(5.0), ne)]
    blog = θ -> twopop_log(spec, edges, θ)
    trace, _ = correct_edges(blog, clamp.(logθ0, llo, lhi), spec, edges, llo, lhi; b.passes, b.runs, T)
    θc = argmin(p -> p.f, trace).θ
    to_signed(θ) = [θ[1]; asinh.(-exp.(θ[2:end]) ./ TWOPOP.ν0)]
    if validate(blog(θc), spec; seeds=8_000_001:8_000_032, T=2T, dtscale=0.01).pass
        verbose && println("  two-population: the correction passes pass the held-out check; no search")
        return twopop_signed(spec, edges, to_signed(θc))
    end
    slo = [log(0.05); fill(asinh(-5.0 / TWOPOP.ν0), ne)]
    shi = [log(200.0); fill(asinh(0.3 / TWOPOP.ν0), ne)]
    s = timed_search(θ -> twopop_signed(spec, edges, θ), spec, clamp.(to_signed(θc), slo .+ 1e-6, shi .- 1e-6), 0.3, slo, shi;
                     b.budget, b.popsize, b.runs, T, maxtime=b.seconds)
    verbose && println("  two-population: search used $(s.evals) evaluations")
    if all(<(0), s.θ[2:end])    # still all heteroclinic, so the log-rate polish applies
        back = [s.θ[1]; log.(-TWOPOP.ν0 .* sinh.(s.θ[2:end]))]
        _, θp = correct_edges(blog, clamp.(back, llo, lhi), spec, edges, llo, lhi; passes=2, runs=32, T=4T, base=200_000)
        return twopop_signed(spec, edges, to_signed(θp))
    end
    return twopop_signed(spec, edges, s.θ)
end
