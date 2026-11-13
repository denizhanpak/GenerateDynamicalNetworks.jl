# CTRNN backend with an edge-by-edge design (the prototype's Phase 5 `:design` regime).
#
# One cell per state, read by argmax. While cell i is on, every other cell j is a one-dimensional noisy
# unit resting W[j, i] = −d below its threshold, and noise fires it at the rate the 1-D first-passage
# time gives, so each edge i → j gets one weight, solved from its target rate p / dwell. The firing
# cell shuts the others off through large negative weights. A two-way pair i ↔ j goes through a hidden
# relay cell for the edge j → i: while j is on the relay rests at −d and is fired by noise like any
# other edge, then drives i on over j's inhibition and holds every other state off. Measured rates then
# correct the edges, one Broyden step per state per pass.

# Levels in units of a cell's stationary noise s = ε√(τ/2): the on level, the inhibition (as a multiple
# of the on level), the sigmoid sharpness (per unit of the smallest s), and each cell's time constant as
# a fraction of the shortest dwell of its state and that state's predecessors.
const CTRNN_ON_SD, CTRNN_OFF_RATIO, CTRNN_SHARPNESS, CTRNN_CELL_FRACTION = 12.0, 1.5, 2.0, 0.25

"""
    CTRNNModel

A continuous-time recurrent network `τ ⊙ dy = (−y + W σ(g(y + b))) dt + τ ⊙ ε dW` per cell, written
`dy = (−y + W σ(g(y + b))) ./ τ dt + ε dW`, with σ the logistic function. The first `visible` cells are
the states, read by the largest `g(yᵢ + bᵢ)`; any after them are hidden relay cells. `α[1]` is the
stiffest rate, which sets the step. Built by `fit(spec, CTRNNDesign())`, or from a parameter vector
with `CTRNNModel(spec, θ)`.
"""
struct CTRNNModel <: AbstractNetworkModel
    P::Matrix{Float64}
    visible::Int
    α::Vector{Float64}
    W::Matrix{Float64}
    b::Vector{Float64}
    τ::Vector{Float64}
    g::Float64
    ε::Float64
end

"""
    CTRNNModel(spec, θ; cap=Inf)

The model with parameters `θ = [vec(W); b; log τ; log g]` over all `N` cells (so `length(θ) == (N+1)²`),
as [`ctrnn_parameters`](@ref) returns them. Throws an `ArgumentError` if the stiffest rate is above `cap`.
"""
function CTRNNModel(spec::TransitionSpec, θ::AbstractVector; cap::Real=Inf)
    n = isqrt(length(θ)) - 1
    (n + 1)^2 == length(θ) || throw(DimensionMismatch("θ has length $(length(θ)), not (N + 1)²"))
    W, b, τ, g = reshape(θ[1:n^2], n, n), θ[n^2+1:n^2+n], exp.(θ[n^2+n+1:n^2+2n]), exp(θ[end])
    rate = max(1.0, g * maximum(abs, W) / 4) / minimum(τ)
    rate > cap && throw(ArgumentError("rate $rate is above the limit $cap"))
    return CTRNNModel(Matrix{Float64}(spec.P), nstates(spec), [rate], Matrix{Float64}(W), Vector{Float64}(b), τ, g, spec.noise)
end

"""
    ctrnn_parameters(m::CTRNNModel) -> Vector{Float64}

`[vec(W); b; log τ; log g]`, the vector [`CTRNNModel`](@ref)`(spec, θ)` reads back.
"""
ctrnn_parameters(m::CTRNNModel) = [vec(m.W); m.b; log.(m.τ); log(m.g)]

ctrnn_drift(y, q, t) = (-y .+ q.W * (1 ./ (1 .+ exp.(-q.g .* (y .+ q.b))))) ./ q.τ

drift_function(::CTRNNModel) = ctrnn_drift
function parameters(m::CTRNNModel)
    n = size(m.W, 1)
    return (; W=SMatrix{n,n}(m.W), b=SVector{n}(m.b), τ=SVector{n}(m.τ), m.g, m.ε, nv=m.visible)
end
initial_state(m::CTRNNModel) = (n = size(m.W, 1); SVector{n}(ntuple(i -> -m.b[i] + (i == 1 ? 4.0 : -4.0) / m.g, n)))
speed(m::CTRNNModel) = maximum(m.α)
in_bounds(::CTRNNModel, y) = maximum(abs, y) < 1000
frame_state(::CTRNNModel, q, y) = argmax(i -> q.g * (y[i] + q.b[i]), 1:q.nv)     # state cells only

# ── The design ──────────────────────────────────────────────────────────────────────────────

"""
    ctrnn_passage_time(d, A, g, τ, ε; n=6000)

Mean time for one cell, resting at `−d` with on level `A`, sharpness `g`, time constant `τ` and noise
`ε`, to reach `A/2`: the 1-D first-passage integral on a grid of `n` points. It is evaluated in plain
floating point, which is fine at the design's levels (`A` = 12 noise SDs) but can overflow to `NaN`
for much higher barriers (it did for one test cell with `A` at 38 noise SDs).
"""
function ctrnn_passage_time(d, A, g, τ, ε; n=6000)
    s = ε * sqrt(τ / 2)
    ys = range(min(-d, 0.0) - 8s, A / 2; length=n + 1)
    Φ = @. (2 / ε^2) * (ys^2 / 2 + d * ys - (A / g) * log1p(exp(g * ys))) / τ
    Φ .-= minimum(Φ)
    h = step(ys)
    inner = cumsum(@. exp(-Φ)) .* h
    start = searchsortedfirst(ys, -d)
    return (2 / ε^2) * sum(exp(Φ[k]) * inner[k] for k in start:n+1) * h
end

"""The resting offset `d` whose passage time is `T`, by bisection on [−A/3, 1.2A] (clamped at the ends)."""
function ctrnn_solve_offset(T, A, g, τ, ε)
    lo, hi = -A / 3, 1.2A
    f(d) = log(ctrnn_passage_time(d, A, g, τ, ε)) - log(T)
    f(lo) >= 0 && return lo
    f(hi) <= 0 && return hi
    return find_zero(f, (lo, hi), Bisection(); xatol=1e-6 * A)
end

"""Two-way pairs i ↔ j with i < j, as the edges j → i that go through a relay cell."""
relayed_edges(P::AbstractMatrix) = [(j, i) for i in axes(P, 1) for j in axes(P, 1) if i < j && P[i, j] > 0 && P[j, i] > 0]

"""
    ctrnn_layout(spec) -> (; n, rel, receiver, τc, A, D, g)

The cells of the design: states `1:n`, then one relay per entry of `rel` (see `relayed_edges`).
`receiver[e]` is the cell noise fires for edge `e`, `τc` the time constants, `A` and `D` each cell's on
level and inhibition, `g` the shared sharpness.
"""
function ctrnn_layout(spec)
    P, target, n = spec.P, spec.τ, size(spec.P, 1)
    rel = relayed_edges(P)
    τs = [CTRNN_CELL_FRACTION * minimum([target[j]; target[findall(>(0), P[:, j])]]) for j in 1:n]
    τc = [τs; [max(τs[j], τs[i]) for (j, i) in rel]]     # a relay outlasts the slower of its two states
    s = spec.noise .* sqrt.(τc ./ 2)
    receiver = [(i, j) in rel ? n + something(findfirst(==((i, j)), rel)) : j for (i, j) in edge_list(P)]
    return (; n, rel, receiver, τc, A=CTRNN_ON_SD .* s, D=CTRNN_OFF_RATIO .* CTRNN_ON_SD .* s, g=CTRNN_SHARPNESS / minimum(s))
end

"""The designed network for edge offsets `d` (one per edge, in `edge_list` order)."""
function ctrnn_design_model(spec, layout, d)
    n, N = layout.n, length(layout.τc)
    W = [a == b ? layout.A[a] : -layout.D[a] for a in 1:N, b in 1:N]
    for (r, (j, i)) in enumerate(layout.rel)
        k = n + r
        for a in 1:n          # relay k drives i on over j's inhibition and, like a firing state, holds every other state off
            W[a, k] = a == i ? layout.D[i] + 0.5layout.A[i] : -layout.D[a]
        end
    end
    for (e, (i, _)) in enumerate(edge_list(spec.P))
        W[layout.receiver[e], i] = -d[e]
    end
    rate = max(1.0, layout.g * maximum(abs, W) / 4) / minimum(layout.τc)
    return CTRNNModel(Matrix{Float64}(spec.P), n, [rate], W, zeros(N), layout.τc, layout.g, spec.noise)
end

"""Starting offsets: each edge's passage time set to its share of the dwell left after the handoffs."""
function ctrnn_seed_offsets(spec, layout)
    edges = edge_list(spec.P)
    delay = zeros(size(spec.P, 1))
    for (e, (i, j)) in enumerate(edges)
        delay[i] += spec.P[i, j] * 2layout.τc[layout.receiver[e]]
    end
    return [let r = layout.receiver[e]
                ctrnn_solve_offset(max(spec.τ[i] - delay[i], 0.3spec.τ[i]) / spec.P[i, j], layout.A[r], layout.g, layout.τc[r], spec.noise)
            end for (e, (i, j)) in enumerate(edges)]
end

"""
    ctrnn_correct_offsets(spec, layout, d0; by_state=true, passes, late_runs, base=100_000,
                          checkpoint=nothing, deadline=Inf, verbose=false) -> d

Correction passes on the edge offsets; returns the offsets after the last update. Each pass simulates
the current design on fresh seeds (32 runs of length 250Στ; the last three passes use `late_runs` runs
at half the step).

- `by_state=true` (10 passes, `late_runs=192`): one Broyden step per state. Inputs are the log of the
  state's total 1-D exit rate and the log rate ratios of its edges; outputs the measured log exit rate
  (1/dwell) and log branch ratios. The Jacobian starts at the identity and is learned from pass to pass.
- `by_state=false` (8 passes, `late_runs=96`): each edge's passage time moves by its own measured rate
  error, at most a factor e per pass.

With `checkpoint` (a file path) the state is saved after every chunk of 16 runs and a later call
resumes from it. With `deadline` (a `time()` value) it throws [`OutOfTime`](@ref) rather than start a
chunk it expects to end later, judging from the last chunk of the same call.
"""
function ctrnn_correct_offsets(spec, layout, d0; by_state::Bool=true, passes::Integer=by_state ? 10 : 8,
                               late_runs::Integer=by_state ? 192 : 96, base::Integer=100_000, checkpoint=nothing,
                               deadline::Real=Inf, verbose::Bool=false)
    n, edges = size(spec.P, 1), edge_list(spec.P)
    T = 250sum(spec.τ)
    params(k) = (layout.A[layout.receiver[k]], layout.g, layout.τc[layout.receiver[k]], spec.noise)
    outs = [[k for (k, (a, _)) in enumerate(edges) if a == i] for i in 1:n]
    st = (; pass=0, d=copy(d0), J=[Matrix{Float64}(I, length(o), length(o)) for o in outs],
          px=[Float64[] for _ in 1:n], py=[Float64[] for _ in 1:n], partial=Any[])
    if checkpoint !== nothing && isfile(checkpoint)
        st = deserialize(checkpoint)
        verbose && println("  resuming after pass ", st.pass, isempty(st.partial) ? "" : ", with $(length(st.partial)) runs of the next")
    end
    d, J, px, py, partial = st.d, st.J, st.px, st.py, st.partial
    per_unit = NaN       # seconds per (run / dtscale), measured afresh in each call
    save(pass_done) = checkpoint !== nothing && serialize(checkpoint, (; pass=pass_done, d, J, px, py, partial, per_unit))
    for pass in st.pass+1:passes
        late = pass > passes - 3
        dts = late ? 0.01 : 0.02
        m = ctrnn_design_model(spec, layout, d)
        seeds = base+1000pass+1:base+1000pass+(late ? late_runs : 32)
        for chunk in Iterators.partition(seeds[length(partial)+1:end], 16)
            if time() + (isnan(per_unit) ? 0.0 : 1.2per_unit * length(chunk) / dts) > deadline
                save(pass - 1)
                throw(OutOfTime())
            end
            started = time()
            append!(partial, simulate(m; T, seeds=collect(chunk), spec.sample, dtscale=dts))
            per_unit = (time() - started) / (length(chunk) / dts)
            save(pass - 1)
        end
        r = [x for x in partial]
        partial = Any[]
        s, meas = score(r, spec), measure(r, n)
        if verbose
            miss = maximum(maximum(abs(meas.counts[i, j] / max(sum(meas.counts[i, :]), 1) - spec.P[i, j]) for j in successors(spec.P, i)) for i in 1:n)
            println("  pass $pass  f $(round(s.f; digits=4))  dwell $(round.(s.dwell; sigdigits=4))  off $(round(s.off; digits=4))  largest branch miss $(round(miss; digits=4))")
        end
        all(isfinite, s.dwell) || break
        if !by_state
            for (k, (i, j)) in enumerate(edges)
                measured = max(meas.counts[i, j], 1) / max(sum(meas.counts[i, :]), 1) / s.dwell[i]
                wanted = spec.P[i, j] / spec.τ[i]
                current = ctrnn_passage_time(d[k], params(k)...)
                d[k] = ctrnn_solve_offset(current * clamp(measured / wanted, exp(-1), exp(1)), params(k)...)
            end
        else
            for i in 1:n
                out, k = outs[i], length(outs[i])
                rate = [1 / ctrnn_passage_time(d[e], params(e)...) for e in out]
                seen = max(sum(meas.counts[i, :]), 1)
                share = [max(meas.counts[i, edges[e][2]] / seen, 1e-3) for e in out]
                want = [spec.P[i, edges[e][2]] for e in out]
                x = [log(sum(rate)); log.(rate[1:k-1]) .- log(rate[k])]
                y = [-log(s.dwell[i]); log.(share[1:k-1]) .- log(share[k])]
                goal = [-log(spec.τ[i]); log.(want[1:k-1]) .- log(want[k])]
                if !isempty(px[i]) && sum(abs2, x .- px[i]) > 0.03^2
                    Δx, Δy = x .- px[i], y .- py[i]
                    J[i] .+= (Δy .- J[i] * Δx) * Δx' ./ sum(abs2, Δx)
                    for a in 1:k, c in 1:k     # keep the learned responses in a sane range
                        J[i][a, c] = a == c ? clamp(J[i][a, c], 0.25, 4.0) : clamp(J[i][a, c], -2.0, 2.0)
                    end
                    cond(J[i]) > 50 && (J[i] .= Diagonal(J[i]))
                end
                px[i], py[i] = x, y
                xn = x .+ clamp.(J[i] \ (goal .- y), -1, 1)
                w = exp.([xn[2:end]; 0.0])
                w ./= sum(w)
                for (a, e) in enumerate(out)
                    d[e] = ctrnn_solve_offset(1 / (exp(xn[1]) * w[a]), params(e)...)
                end
            end
        end
        save(pass)
    end
    return d
end

"""
    CTRNNDesign(; by_state=true, passes=10, late_runs=192, base=100_000)

The CTRNN backend's edge-by-edge design: one cell per state plus one relay cell per two-way pair, each
edge's weight solved from a 1-D first-passage time, then correction passes from measured rates (see
`ctrnn_correct_offsets`; `by_state=false` uses 8 passes and 96 late runs by default). Seeds start at
`base`, so different `base`s give independent fits.

    fit(spec, CTRNNDesign(); checkpoint=nothing, deadline=Inf, verbose=false) -> CTRNNModel
"""
struct CTRNNDesign <: AbstractBackend
    by_state::Bool
    passes::Int
    late_runs::Int
    base::Int
end
CTRNNDesign(; by_state::Bool=true, passes::Integer=by_state ? 10 : 8, late_runs::Integer=by_state ? 192 : 96,
            base::Integer=100_000) = CTRNNDesign(by_state, passes, late_runs, base)

function fit(spec::TransitionSpec, b::CTRNNDesign; checkpoint=nothing, deadline::Real=Inf, verbose::Bool=false)
    layout = ctrnn_layout(spec)
    d = ctrnn_correct_offsets(spec, layout, ctrnn_seed_offsets(spec, layout); b.by_state, b.passes, b.late_runs, b.base,
                              checkpoint, deadline, verbose)
    return ctrnn_design_model(spec, layout, d)
end
