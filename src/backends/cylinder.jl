# Cylinder backend (Ashwin & Postlethwaite 2013): a phase p whose integer points 1, …, n are the states and one
# cell y per edge, one global speed, a rate per edge, one shared noise, no hidden states. The construction's
# steepness k (k_α = k_ω) and L_ω, and the decoder's windows, are options; the defaults are the prototype's.

"""The eigenvalue at state i along the edge to j (the paper's e_{ij}), solved for L_α."""
cylinder_Lα(e, i, j, k, Lω) = e + 9 / 16 + Lω * sech(k * (i - j))^2

function check_cylinder(k, Lω, pwindow, ywindow)
    k > 0 && Lω > 0 || throw(ArgumentError("k and Lω must be positive"))
    0 < pwindow <= 0.5 || throw(ArgumentError("pwindow must be in (0, 0.5]"))
    ywindow > 0 || throw(ArgumentError("ywindow must be positive"))
    return nothing
end

"""
    CylinderModel{N}

Ashwin & Postlethwaite's cylinder network on a graph with `N` edges: edge cells `y` and a phase `p`, at speed
`s`, with `Lα[m]` for edge `m` (`from[m] → to[m]`), steepness `k` and `Lω`. Read as state j when
|p − j| < `pwindow` and every |y| < `ywindow`; otherwise the last state holds. Built by
`fit(spec, Cylinder())` or `CylinderModel(spec, θ)`.
"""
struct CylinderModel{N} <: AbstractNetworkModel
    P::Matrix{Float64}
    visible::Int
    s::Float64
    Lα::NTuple{N,Float64}
    from::NTuple{N,Float64}
    to::NTuple{N,Float64}
    ε::Float64
    k::Float64
    Lω::Float64
    pwindow::Float64
    ywindow::Float64
    function CylinderModel{N}(P, visible, s, Lα, from, to, ε, k, Lω, pwindow, ywindow) where {N}
        check_cylinder(k, Lω, pwindow, ywindow)
        return new{N}(P, visible, s, Lα, from, to, ε, k, Lω, pwindow, ywindow)
    end
end

"""
    CylinderModel(spec, θ; k=2.0, Lω=1.5625, pwindow=0.25, ywindow=0.5)

The model with `θ = [log s; log eᵢⱼ for each edge in edge_list order]`, eᵢⱼ the rate at which state i
leaves along its edge to j.
"""
function CylinderModel(spec::TransitionSpec, θ::AbstractVector; k::Real=2.0, Lω::Real=1.5625, pwindow::Real=0.25,
                       ywindow::Real=0.5)
    edges = edge_list(spec.P)
    N = length(edges)
    length(θ) == 1 + N || throw(ArgumentError("θ must hold 1 + $N values (speed, edges), not $(length(θ))"))
    e = exp.(θ[2:1+N])
    Lα = ntuple(m -> cylinder_Lα(e[m], edges[m]..., float(k), float(Lω)), N)
    return CylinderModel{N}(Matrix{Float64}(spec.P), nstates(spec), exp(θ[1]), Lα, ntuple(m -> float(edges[m][1]), N),
                            ntuple(m -> float(edges[m][2]), N), float(spec.noise), float(k), float(Lω), float(pwindow),
                            float(ywindow))
end

function cylinder_drift(z::SVector{M}, q, t) where {M}
    N = M - 1
    p, y2 = z[M], sum(i -> z[i]^2, 1:N)
    dy = ntuple(Val(N)) do j
        G = (z[j]^2 - 5 / 4)^2 - 1 - q.Lα[j] * sech(q.k * (p - q.from[j]))^2 + q.Lω * sech(q.k * (p - q.to[j]))^2 + y2 - z[j]^2
        -q.s * z[j] * G
    end
    F = sum(j -> z[j]^2 * tanh(q.to[j] - p), 1:N)
    return SVector{M}(dy..., q.s * (-sinpi(2p) + F))
end

drift_function(::CylinderModel) = cylinder_drift
parameters(m::CylinderModel) = (; m.s, m.Lα, m.from, m.to, m.ε, m.k, m.Lω, m.visible, m.pwindow, m.ywindow)
initial_state(m::CylinderModel{N}) where {N} = SVector{N + 1}(ntuple(i -> i <= N ? m.ε : 1.0, N + 1))
speed(m::CylinderModel) = m.s
function frame_state(::CylinderModel, q, z)
    M = length(z)
    p = z[M]
    abs(p) < q.visible + 1 || return 0          # off the states (or not finite)
    j = round(Int, p)
    return 1 <= j <= q.visible && abs(p - j) < q.pwindow && maximum(i -> abs(z[i]), 1:M-1) < q.ywindow ? j : 0
end
class_start(::CylinderModel{N}) where {N} = SVector{N + 1}(ntuple(i -> i <= N ? 1e-3 : 1.0, N + 1))

"""The cylinder fit's start and box: the edge seed ([`edge_seed_rates`](@ref)) scaled by the contraction Lω + 9/16."""
function cylinder_start(spec::TransitionSpec, Lω)
    edges = edge_list(spec.P)
    ne = length(edges)
    contract = Lω + 9 / 16
    r = edge_seed_rates(spec, edges)
    θ0 = [log(r.α / contract); r.logs .+ log(contract)]
    lower = [log(0.05); fill(log(1e-3), ne)]
    upper = [log(200.0); fill(log(8.0), ne)]
    return (; θ0, lower, upper, r.seeded)
end

"""
    Cylinder(; k=2.0, Lω=1.5625, pwindow=0.25, ywindow=0.5, passes=4, runs=16, T=nothing, budget=500, popsize=12,
             seconds=300.0)

The cylinder backend's fit (Ashwin & Postlethwaite 2013): one global speed and a rate per edge
([`CylinderModel`](@ref)), with steepness `k`, `Lω` and the decoder windows `pwindow` and `ywindow` fixed,
one shared noise, no hidden states. A seed from the Horchler dwell design ([`edge_seed_rates`](@ref)),
`passes` correction passes of `runs` runs of length `T` (default 125Στ), and if the best of those fails a
check on 32 held-out runs, a CMA-ES search over the speed and edge rates of at most `budget` evaluations or
`seconds` seconds, then two correction passes as a polish. (The prototype stopped at 420 s by default.) A
binding time limit makes the result depend on the machine's speed; `seconds=Inf` limits the search by
`budget` alone.

    fit(spec, Cylinder(); verbose=false) -> CylinderModel

!!! warning "Status: experimental"
    Fails the acceptance protocol on half or more of the benchmark cases; see the Results page of the documentation.
"""
struct Cylinder <: AbstractBackend
    k::Float64
    Lω::Float64
    pwindow::Float64
    ywindow::Float64
    passes::Int
    runs::Int
    T::Union{Float64,Nothing}
    budget::Int
    popsize::Int
    seconds::Float64
    function Cylinder(k::Real, Lω::Real, pwindow::Real, ywindow::Real, passes::Integer, runs::Integer,
                      T::Union{Real,Nothing}, budget::Integer, popsize::Integer, seconds::Real)
        check_cylinder(k, Lω, pwindow, ywindow)
        passes >= 1 || throw(ArgumentError("passes must be at least 1"))
        return new(k, Lω, pwindow, ywindow, passes, runs, T === nothing ? nothing : float(T), budget, popsize, seconds)
    end
end
Cylinder(; k::Real=2.0, Lω::Real=1.5625, pwindow::Real=0.25, ywindow::Real=0.5, passes::Integer=4, runs::Integer=16,
         T::Union{Real,Nothing}=nothing, budget::Integer=500, popsize::Integer=12, seconds::Real=300.0) =
    Cylinder(k, Lω, pwindow, ywindow, passes, runs, T, budget, popsize, seconds)

function fit(spec::TransitionSpec, b::Cylinder; verbose::Bool=false)
    T = b.T === nothing ? 125sum(spec.τ) : b.T
    st = cylinder_start(spec, b.Lω)
    verbose && !st.seeded && println("  cylinder: the dwell design has no solution; starting from a rough guess")
    build(θ) = CylinderModel(spec, θ; b.k, b.Lω, b.pwindow, b.ywindow)
    θ = edge_rate_fit(build, spec, st.θ0, st.lower, st.upper; b.passes, b.runs, T, b.budget, b.popsize, b.seconds,
                      label="cylinder", verbose)
    return build(θ)
end
