# Out-rate fitting shared by the synchrony and Hopfield backends: the rule-of-thumb start and the
# per-state Broyden correction on simulated dwell times and branch shares.

"""
    rough_rates(spec) -> (rough, fast)

The rule-of-thumb out-rates, one per edge in `edge_list` order, `log(1/ε)/τᵢ · √(Pᵢⱼ · out-degree of i)`,
and 4× the largest of them, used as both the transit rate and the contraction.
"""
function rough_rates(spec)
    n, edges = nstates(spec), edge_list(spec.P)
    outs = [count(e -> e[1] == i, edges) for i in 1:n]
    law = log(1 / spec.noise)
    rough = [law / spec.τ[i] * sqrt(spec.P[i, j] * outs[i]) for (i, j) in edges]
    return rough, 4maximum(rough)
end

"""
    initial_response(k, law=log(10))

A state's starting Broyden matrix over its `k` out-edges, from inputs `asinh(λₑ/c)` to outputs
`[−log dwell; log share ratios]`: a faster exit along any edge shortens the dwell and shifts the share
toward that edge. The share response scales with `law = ln(1/ε)`: at low noise the larger rate wins
almost every time, so shares are far steeper in the rate ratio.
"""
function initial_response(k, law=log(10))
    k == 1 && return ones(1, 1)
    J = zeros(k, k)
    J[1, :] .= 1 / k
    b = 2.0law / log(10)
    for a in 1:k-1
        J[a+1, a], J[a+1, k] = b, -b
    end
    return J
end

"""
    correct_rates(spec, build, out0; passes=12, base=100_000, response=nothing, cap=Inf, T=250Στ,
                  checkpoint=nothing, deadline=Inf, verbose=false) -> (; out, trace)

Correction passes on out-rates (one per edge, in `edge_list` order) for a model `build(out)`. Each pass
simulates the model and takes, per state, one Broyden step on `x = asinh.(λₑ/c)` over its out-edges
against `y = [−log dwell; log branch ratios]`, with `c` 0.05× the smallest starting |rate|. `asinh`
behaves like `log` for large rates and is linear through zero, so a rate may turn negative (the
excitable side). The Jacobian starts from `response(o, out, c)` (or [`initial_response`](@ref)) and is
learned between passes, skipping states with a branch that saw no visits. Passes use 32 runs at
`dtscale = 0.02`; the last 3 use 96 runs at 0.01 and steps of at most 0.3 in `x`. `cap` bounds every
rate's size on both sides. The last update is returned, with a trace of every pass.

Runs go in chunks of 16. With `checkpoint` (a file path) the state is saved after every chunk and a
later call resumes from it. With `deadline` (a `time()` value) it throws [`OutOfTime`](@ref) rather than
start a chunk expected to end after it.
"""
function correct_rates(spec, build, out0; passes::Integer=12, base::Integer=100_000, response=nothing, cap::Real=Inf,
                       T::Real=250sum(spec.τ), checkpoint=nothing, deadline::Real=Inf, verbose::Bool=false)
    n, edges = nstates(spec), edge_list(spec.P)
    outs = [[e for (e, (a, _)) in enumerate(edges) if a == i] for i in 1:n]
    law = log(1 / spec.noise)
    out = copy(out0)
    c = 0.05minimum(abs.(out))
    start_J(o) = response === nothing ? initial_response(length(o), law) : response(o, out, c)
    J = [start_J(o) for o in outs]
    px, py = [Float64[] for _ in 1:n], [Float64[] for _ in 1:n]
    trace, partial, done = Any[], Any[], 0
    if checkpoint !== nothing && isfile(checkpoint)
        st = deserialize(checkpoint)
        # the asinh scale is kept too: a changed scale would shift every x and break the Broyden update
        out, J, px, py, trace, partial, done, c = st.out, st.J, st.px, st.py, st.trace, st.partial, st.done, st.c
        verbose && println("  resuming after pass $done", isempty(partial) ? "" : ", with $(length(partial)) runs of the next")
    end
    save() = checkpoint !== nothing && serialize(checkpoint, (; out, J, px, py, trace, partial, done, c))
    per_unit = NaN       # seconds per (run / dtscale), measured afresh in each call
    for pass in done+1:passes
        late = pass > passes - 3
        dts = late ? 0.01 : 0.02
        m = build(out)
        seeds = base+1000pass+1:base+1000pass+(late ? 96 : 32)
        for chunk in Iterators.partition(seeds[length(partial)+1:end], 16)
            if time() + (isnan(per_unit) ? 0.0 : 1.2per_unit * length(chunk) / dts) > deadline
                save()
                throw(OutOfTime())
            end
            started = time()
            append!(partial, simulate(m; T, seeds=collect(chunk), spec.sample, dtscale=dts))
            per_unit = (time() - started) / (length(chunk) / dts)
            save()
        end
        r = [x for x in partial]
        partial = Any[]
        s, meas = score(r, spec), measure(r, n)
        miss = maximum(maximum(abs(meas.counts[i, j] / max(sum(meas.counts[i, :]), 1) - spec.P[i, j]) for j in successors(spec.P, i)) for i in 1:n)
        verbose && println("  pass $pass  f $(round(s.f; digits=4))  dwell $(round.(s.dwell; sigdigits=4))  off $(round(s.off; digits=4))  largest branch miss $(round(miss; digits=4))")
        push!(trace, (; pass, out=copy(out), s.f, s.dwell, s.off, miss))
        for i in 1:n
            isfinite(s.dwell[i]) || continue                # a state not visited this pass keeps its rates
            o, k = outs[i], length(outs[i])
            seen = max(sum(meas.counts[i, :]), 1)
            share = [max(meas.counts[i, edges[e][2]] / seen, 1e-3) for e in o]
            saturated = any(e -> meas.counts[i, edges[e][2]] == 0, o)
            want = [spec.P[i, edges[e][2]] for e in o]
            x = asinh.(out[o] ./ c)
            y = [-log(s.dwell[i]); log.(share[1:k-1]) .- log(share[k])]
            goal = [-log(spec.τ[i]); log.(want[1:k-1]) .- log(want[k])]
            # a branch with no visits gives no slope worth learning from, in this pass or the next
            if !saturated && !isempty(px[i]) && all(isfinite, py[i]) && sum(abs2, x .- px[i]) > 0.02^2
                Δx, Δy = x .- px[i], y .- py[i]
                J[i] .+= (Δy .- J[i] * Δx) * Δx' ./ sum(abs2, Δx)
                J[i] .= clamp.(J[i], -10.0, 10.0)
                (cond(J[i]) > 100 || all(iszero, J[i][1, :])) && (J[i] = start_J(o))
            end
            px[i], py[i] = x, saturated ? fill(NaN, k) : y
            step = late ? 0.3 : 1.0
            out[o] = clamp.(c .* sinh.(x .+ clamp.(J[i] \ (goal .- y), -step, step)), -cap, cap)
        end
        done = pass
        save()
    end
    return (; out, trace)
end
