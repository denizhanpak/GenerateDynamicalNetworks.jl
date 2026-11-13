# Pilot: an edge-by-edge CTRNN design, one cell per state, read by argmax as agreed.
#
# While cell i is on, every other cell j is a one-dimensional noisy unit sitting at W[j, i] below its
# threshold. Noise fires it at a rate that the first-passage formula for one dimension gives exactly,
# so each edge i → j gets one weight, W[j, i] = −d, solved from its target rate p / dwell. Rates of the
# successors add, which gives the dwell and the branch probabilities together. The firing cell shuts
# the others off through large negative weights. Measured rates then correct each edge separately.
#
# A two-way pair i ↔ j can't be built that way: W[i, j] would have to let j's noise fire i and also
# let j shut i off. So for each pair (i < j) the edge j → i goes through a hidden relay cell k. While j
# is on, k rests at −d below threshold and noise fires it like any other edge; k then drives i on,
# over j's inhibition, and holds every other state off, so a near-tie with j's other successors can't
# fire both; i in turn shuts k off. W[i, j] stays strongly negative. The
# decoder reads state cells only, so the readout goes straight from j to i. Graphs with no two-way
# edge get no relay and exactly the model they had before.
#
# Run from notebooks/ with
#   julia --project=. -t 8 ../benchmarks/phase5/ctrnn_design.jl [noise] [case names...]
isdefined(@__MODULE__, :CTRNNModel) || include(joinpath(@__DIR__, "backends.jl"))
using Roots, LinearAlgebra, Serialization

const ON_SD, OFF_RATIO, SHARPNESS, CELL_FRACTION = 12.0, 1.5, 2.0, 0.25

"""Mean time for one cell to reach the top of its bistable region from its resting level −d."""
function passage_time(d, A, g, τ, ε; n=6000)
    s = ε * sqrt(τ / 2)
    ys = range(min(-d, 0.0) - 8s, A / 2; length=n + 1)
    Φ = @. (2 / ε^2) * (ys^2 / 2 + d * ys - (A / g) * log1p(exp(g * ys))) / τ
    Φ .-= minimum(Φ)
    h = step(ys)
    inner = cumsum(@. exp(-Φ)) .* h
    start = searchsortedfirst(ys, -d)
    return (2 / ε^2) * sum(exp(Φ[k]) * inner[k] for k in start:n+1) * h
end

function solve_offset(T, A, g, τ, ε)
    lo, hi = -A / 3, 1.2A
    f(d) = log(passage_time(d, A, g, τ, ε)) - log(T)
    f(lo) >= 0 && return lo
    f(hi) <= 0 && return hi
    return find_zero(f, (lo, hi), Bisection(); xatol=1e-6 * A)
end

"""Two-way pairs i ↔ j with i < j, as the edges j → i that go through a relay cell."""
relayed(P) = [(j, i) for i in axes(P, 1) for j in axes(P, 1) if i < j && P[i, j] > 0 && P[j, i] > 0]

"""
Cell time constants and the levels in units of each cell's stationary noise. Cells 1:n are the states
and n+1:N the relays. `receiver[e]` is the cell that noise fires for edge e.
"""
function design_layout(spec)
    P, target, n = spec.P, spec.τ, size(spec.P, 1)
    rel = relayed(P)
    τs = [CELL_FRACTION * minimum([target[j]; target[findall(>(0), P[:, j])]]) for j in 1:n]
    τc = [τs; [max(τs[j], τs[i]) for (j, i) in rel]]     # a relay outlasts the slower of its two states
    s = spec.ε₀ .* sqrt.(τc ./ 2)
    receiver = [(i, j) in rel ? n + findfirst(==((i, j)), rel) : j for (i, j) in edge_list(P)]
    return (; n, rel, receiver, τc, A=ON_SD .* s, D=OFF_RATIO .* ON_SD .* s, g=SHARPNESS / minimum(s))
end

function design_model(spec, layout, d)
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
    return CTRNNModel(float(spec.P), n, [rate], W, zeros(N), layout.τc, layout.g, float(spec.ε₀))
end

function seed_offsets(spec, layout)
    edges = edge_list(spec.P)
    delay = zeros(size(spec.P, 1))
    for (e, (i, j)) in enumerate(edges)
        delay[i] += spec.P[i, j] * 2layout.τc[layout.receiver[e]]
    end
    return [let r = layout.receiver[e]
                solve_offset(max(spec.τ[i] - delay[i], 0.3spec.τ[i]) / spec.P[i, j], layout.A[r], layout.g, layout.τc[r], spec.ε₀)
            end for (e, (i, j)) in enumerate(edges)]
end

"""
Correction passes. The plain rule (`by_state=false`: 8 passes, the last 3 with 96 runs; it made the six
one-way suite fits) moves each edge's 1-D passage time by its measured rate error. That under-corrects
when part of a dwell is a fixed handoff rather than waiting for noise, as with relays, and it ignores
how a state's edges interact.

The default takes one Broyden step per state. Its inputs are the log of the state's total 1-D exit rate
and the log rate ratios of its edges; its outputs are the measured log exit rate (1/dwell) and log
branch ratios; the Jacobian between them starts at the identity and is learned from pass to pass. A race
between near-deterministic clocks makes the branch shares depend on the total rate as well as on the
ratios, which separate dwell and share steps can't see. 10 passes, the last 3 with 192 runs, and the
result of the last update is returned. With `checkpoint` the state is saved after every pass and a rerun
resumes from it. Each pass runs in chunks of 16 runs, and finished chunks are saved too. It throws
`OutOfTime` rather than start a chunk it expects to end after `deadline` (a `time()` value), judging
from the last chunk of the same batch; the first chunk of a batch always runs.
"""
function correct_offsets(spec, layout, d0; by_state=true, passes=by_state ? 10 : 8, late_runs=by_state ? 192 : 96,
                         base=100_000, checkpoint=nothing, deadline=Inf)
    n, edges = size(spec.P, 1), edge_list(spec.P)
    T = 250sum(spec.τ)
    params(k) = (layout.A[layout.receiver[k]], layout.g, layout.τc[layout.receiver[k]], spec.ε₀)
    outs = [[k for (k, (a, _)) in enumerate(edges) if a == i] for i in 1:n]
    st = (; pass=0, d=copy(d0), J=[Matrix{Float64}(I, length(o), length(o)) for o in outs],
          px=[Float64[] for _ in 1:n], py=[Float64[] for _ in 1:n])
    if checkpoint !== nothing && isfile(checkpoint)
        st = deserialize(checkpoint)
        println("  resuming after pass ", st.pass, hasproperty(st, :partial) && !isempty(st.partial) ? ", with $(length(st.partial)) runs of the next" : "")
    end
    d, J, px, py = st.d, st.J, st.px, st.py
    partial = hasproperty(st, :partial) ? st.partial : Any[]          # finished runs of the pass in progress
    per_unit = NaN       # seconds per (run / dtscale), measured afresh in each batch: one slow chunk once stalled every later batch
    save(pass_done) = checkpoint !== nothing && serialize(checkpoint, (; pass=pass_done, d, J, px, py, partial, per_unit))
    for pass in st.pass+1:passes
        late = pass > passes - 3
        dts = late ? 0.01 : 0.02
        m = design_model(spec, layout, d)
        seeds = base+1000pass+1:base+1000pass+(late ? late_runs : 32)
        for chunk in Iterators.partition(seeds[length(partial)+1:end], 16)
            if time() + (isnan(per_unit) ? 0.0 : 1.2per_unit * length(chunk) / dts) > deadline
                save(pass - 1)
                throw(OutOfTime())
            end
            started = time()
            append!(partial, simulate_frames(m; T, seeds=collect(chunk), spec.sample, dtscale=dts))
            per_unit = (time() - started) / (length(chunk) / dts)
            save(pass - 1)
        end
        r = [x for x in partial]
        partial = Any[]
        s, meas = score(r, spec), measure(r, n)
        miss = [maximum(abs(meas.counts[i, j] / max(sum(meas.counts[i, :]), 1) - spec.P[i, j]) for j in successors(spec.P, i)) for i in 1:n]
        @printf("  pass %d  f %.4f  dwell %s  off %.4f  largest branch miss %.4f\n", pass, s.f, round.(s.dwell; sigdigits=4), s.off, maximum(miss))
        flush(stdout)
        all(isfinite, s.dwell) || break
        if !by_state
            for (k, (i, j)) in enumerate(edges)
                measured = max(meas.counts[i, j], 1) / max(sum(meas.counts[i, :]), 1) / s.dwell[i]
                wanted = spec.P[i, j] / spec.τ[i]
                current = passage_time(d[k], params(k)...)
                d[k] = solve_offset(current * clamp(measured / wanted, exp(-1), exp(1)), params(k)...)
            end
        else
            for i in 1:n
                out, k = outs[i], length(outs[i])
                rate = [1 / passage_time(d[e], params(e)...) for e in out]
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
                    d[e] = solve_offset(1 / (exp(xn[1]) * w[a]), params(e)...)
                end
            end
            println("    learned responses (dwell; branch) per state: ", join((join(round.(diag(Ji); digits=2), "; ") for Ji in J), " | "))
        end
        save(pass)
    end
    return d
end

"""The `:design` regime for the suite: seed and correction passes only, with no search."""
function fit_ctrnn_design(spec; checkpoint=nothing, deadline=Inf, kw...)
    layout = design_layout(spec)
    d = correct_offsets(spec, layout, seed_offsets(spec, layout); checkpoint, deadline, kw...)
    model = design_model(spec, layout, d)
    θ = [vec(model.W); model.b; log.(model.τ); log(model.g)]
    return (; model, θ, stage=:correction, evals=0, seeded=true, dim=length(d))
end

# The pilot itself, when this file is run rather than included.
if abspath(PROGRAM_FILE) == @__FILE__
const OUT = joinpath(@__DIR__, "results")
const CSV_FILE = joinpath(OUT, "ctrnn_design.csv")
noise = isempty(ARGS) ? 0.1 : parse(Float64, ARGS[1])
cases = length(ARGS) >= 2 ? ARGS[2:end] : ["worm ring", "foraging"]

for name in cases
    _, P, τ = only(c for c in NAMED if c[1] == name)
    spec = (; P, τ, form=:u, ε₀=noise, sample=name == "worm ring" ? WORM_SPEC.sample : minimum(τ) / 10)
    println("\n", name, " at noise ", noise)
    t = @elapsed begin
        layout = design_layout(spec)
        d0 = seed_offsets(spec, layout)
        d = correct_offsets(spec, layout, d0)
        model = design_model(spec, layout, d)
        r = validate(model, spec; T=250sum(τ))
        again = validate(model, spec; T=250sum(τ), seeds=9_200_001:9_200_064)
    end
    @printf("  cell τ %s, gain %.3g, step-rule rate %.0f\n", round.(layout.τc; sigdigits=3), layout.g, model.α[1])
    println("  edge offsets, in noise SDs of the receiving cell: ",
            [round(d[k] / (noise * sqrt(layout.τc[layout.receiver[k]] / 2)); digits=2) for k in eachindex(d)])
    show_result("ctrnn design $name ε=$noise", r)
    for (b, c) in zip(r.branch, again.branch)
        @printf("    branch %d→%d designed %.2f measured %.3f %s, repeat %.3f %s\n", b[1], b[2], b[3], b[4], b[5] ? "ok" : "outside CI", c[4], c[5] ? "ok" : "outside CI")
    end
    cv = [std(x) / mean(x) for x in measure(simulate_frames(model; T=250sum(τ), seeds=9_000_001:9_000_016, spec.sample, dtscale=0.01), size(P, 1)).samples]
    println("  bout CV ", round.(cv; digits=2), "   repeat dwell errors ", round.(again.err; digits=3), "  repeat pass ", again.pass, ", failed runs ", r.failed)
    @printf("  %.0f s\n", t)
    writedlm(joinpath(OUT, "theta_ctrnn_design_$(replace(name, r"[^A-Za-z0-9]+" => "_"))_$noise.csv"),
             [vec(model.W); model.b; log.(model.τ); log(model.g)], ',')
    open(CSV_FILE, "a") do io
        filesize(CSV_FILE) == 0 && println(io, "case,noise,worst_err,rms,off,dwell_ok,branches_ok,repeat_branches_ok,pass,rate,seconds")
        println(io, join(("\"$name\"", noise, maximum(abs, r.err), r.rms, r.off, r.dwell_ok, r.branch_ok, again.branch_ok, r.pass && again.pass, model.α[1], t), ","))
    end
    flush(stdout)
end
end
