# Measurement and the acceptance rule.

"""
    measure(runs, n) -> (; samples, counts, dwell, sem)

Dwell statistics pooled over runs from [`simulate`](@ref). Each run drops its first visit (it
starts part-way through) and its last (it has no end). `samples[i]` are the bout lengths in state
`i`, `counts[i, j]` the transitions from `i` to `j`, `dwell` and `sem` the mean bout length and its
standard error (`NaN` with too few bouts).
"""
function measure(runs, n::Integer)
    samples = [Float64[] for _ in 1:n]
    counts = zeros(Int, n, n)
    for v in runs
        s, t = v.state[2:end], v.t[2:end]
        for k in 1:length(s)-1
            push!(samples[s[k]], t[k+1] - t[k])
            counts[s[k], s[k+1]] += 1
        end
    end
    return (; samples, counts, dwell=[isempty(s) ? NaN : mean(s) for s in samples],
              sem=[length(s) < 2 ? NaN : std(s) / sqrt(length(s)) for s in samples])
end

"""
    off_graph(meas, P) -> Float64

The share of measured transitions that are not edges of `P`.
"""
off_graph(meas, P::AbstractMatrix) = sum(meas.counts[c] for c in CartesianIndices(P) if P[c] == 0; init=0) / max(sum(meas.counts), 1)

"""
    verdict(meas, P, τ; tol=0.05) -> (; err, branch, off, pass)

The acceptance rule on one set of runs: every mean dwell within `tol` of `τ` (`err` holds the
relative errors), every branch of a state with two or more successors inside the 95% binomial
interval around its designed probability (`branch` holds one `(; from, to, designed, measured, ok)`
per such edge), and fewer than 1% of transitions off the graph (`off`).
"""
function verdict(meas, P::AbstractMatrix, τ::AbstractVector; tol::Real=0.05)
    n = size(P, 1)
    err = meas.dwell ./ τ .- 1
    branch = @NamedTuple{from::Int, to::Int, designed::Float64, measured::Float64, ok::Bool}[]
    for i in 1:n, j in 1:n
        P[i, j] > 0 && count(>(0), P[i, :]) >= 2 || continue
        total = sum(meas.counts[i, :])
        f = meas.counts[i, j] / max(total, 1)
        half = 1.96sqrt(P[i, j] * (1 - P[i, j]) / max(total, 1))
        push!(branch, (; from=i, to=j, designed=P[i, j], measured=f, ok=abs(f - P[i, j]) <= half))
    end
    off = off_graph(meas, P)
    return (; err, branch, off, pass=all(abs.(err) .<= tol) && all(b -> b.ok, branch) && off < 0.01)
end

"""
    score(runs, spec; off_free=0.005, off_weight=5.0) -> (; f, dwell, off, branch_term, dwell_term)

The fitting objective: Σ log(measured dwell / τ)², each term capped at 25 (35 for a state never
visited), plus Σ log(measured / designed share)² over the branches of states with two or more
successors, plus `off_weight` per unit of off-graph share above `off_free`. A set with a failed run
scores 1000. `spec` needs fields `P` and `τ`.
"""
function score(runs, spec; off_free::Real=0.005, off_weight::Real=5.0)
    n = size(spec.P, 1)
    any(r.failed for r in runs) && return (; f=1000.0, dwell=fill(NaN, n), off=1.0, branch_term=NaN, dwell_term=NaN)
    meas = measure(runs, n)
    dwell_term = sum(isfinite(meas.dwell[i]) ? min(log(meas.dwell[i] / spec.τ[i])^2, 25.0) : 35.0 for i in 1:n)
    branch_term = 0.0
    for i in 1:n
        s = findall(>(0), spec.P[i, :])
        length(s) >= 2 || continue
        total = max(sum(meas.counts[i, :]), 1)
        branch_term += sum(log(max(meas.counts[i, j] / total, 1e-3) / spec.P[i, j])^2 for j in s)
    end
    off = off_graph(meas, spec.P)
    return (; f=dwell_term + branch_term + off_weight * max(off - off_free, 0.0), dwell=meas.dwell, off, branch_term, dwell_term)
end

"""One acceptance check: a model on one set of held-out runs. Built by [`validate`](@ref)."""
struct Check
    dwell::Vector{Float64}
    sem::Vector{Float64}
    err::Vector{Float64}
    rms::Float64
    off::Float64
    branch::Vector{@NamedTuple{from::Int, to::Int, designed::Float64, measured::Float64, ok::Bool}}
    counts::Matrix{Int}
    failed::Int
    dwell_ok::Bool
    branch_ok::Bool
    pass::Bool
end

function Base.show(io::IO, c::Check)
    print(io, "Check(dwell ", join((@sprintf("%.3f", d) for d in c.dwell), " "),
          ", err% ", join((@sprintf("%+.1f", 100e) for e in c.err), " "),
          @sprintf(", rms %.2f%%, off %.2f%%", 100c.rms, 100c.off),
          ", dwell ", c.dwell_ok ? "ok" : "NO", ", branches ", c.branch_ok ? "ok" : "NO",
          c.failed > 0 ? ", $(c.failed) failed runs" : "", ": ", c.pass ? "PASS" : "fail", ")")
end

"""
    validate(m, spec; seeds=9_000_001:9_000_064, T=250sum(spec.τ), dtscale=0.01, tol=0.05) -> Check

Simulate `m` on the given seeds and apply [`verdict`](@ref). The check passes if the verdict does
and no run failed. `rms` is the root mean square of log(measured dwell / τ).
"""
function validate(m::AbstractNetworkModel, spec::TransitionSpec; seeds=9_000_001:9_000_064, T::Real=250sum(spec.τ),
                  dtscale::Real=0.01, tol::Real=0.05)
    runs = simulate(m, spec; T, seeds, dtscale)
    return check_runs(runs, spec; tol)
end

function check_runs(runs, spec; tol::Real=0.05)
    failed = count(r -> r.failed, runs)
    meas = measure(runs, size(spec.P, 1))
    v = verdict(meas, spec.P, spec.τ; tol)
    dwell_ok, branch_ok = all(abs.(v.err) .<= tol), all(b -> b.ok, v.branch)
    return Check(meas.dwell, meas.sem, v.err, sqrt(mean(log.(meas.dwell ./ spec.τ) .^ 2)), v.off, v.branch, meas.counts,
                 failed, dwell_ok, branch_ok, v.pass && failed == 0)
end

"""The three checks of the acceptance protocol and whether all pass. Built by [`accept`](@ref)."""
struct Acceptance
    set1::Check
    set2::Check
    half_step::Check
    pass::Bool
end

function Base.show(io::IO, a::Acceptance)
    print(io, "Acceptance: ", a.pass ? "PASS" : "fail", "\n  set 1:     ", a.set1, "\n  set 2:     ", a.set2,
          "\n  half step: ", a.half_step)
end

"""
    accept(m, spec; T=250sum(spec.τ), set1=9_000_001:9_000_064, set2=9_200_001:9_200_064,
           dtscale=0.01, tol=0.05) -> Acceptance

The acceptance protocol: [`validate`](@ref) on two sets of held-out seeds, and on the first set
again at half the step. A model is accepted if it passes all three.
"""
function accept(m::AbstractNetworkModel, spec::TransitionSpec; T::Real=250sum(spec.τ), set1=9_000_001:9_000_064,
                set2=9_200_001:9_200_064, dtscale::Real=0.01, tol::Real=0.05)
    a = validate(m, spec; seeds=set1, T, dtscale, tol)
    b = validate(m, spec; seeds=set2, T, dtscale, tol)
    c = validate(m, spec; seeds=set1, T, dtscale=dtscale / 2, tol)
    return Acceptance(a, b, c, a.pass && b.pass && c.pass)
end
