# The dynamical classifier: noise-free runs started from points along a noisy run.
#   settles on a stable equilibrium                 → excitable
#   stalls at a saddle, or cycles that keep growing → heteroclinic
#   repeating cycle, peak/slowest speed < ratio     → limit cycle
#   repeating cycle, peak/slowest speed ≥ ratio     → ghost
#   starts that disagree, or nothing recognisable   → unknown

zero_noise(z, q, t) = zero(z)
noise_eta(z, q, t) = q.η * ones(typeof(z))

"""Jacobian of `f(u, q, 0)` at `u` by central differences."""
function jac(f, q, u)
    h = 1e-6
    bump(i) = typeof(u)(ntuple(k -> k == i ? h : 0.0, length(u)))
    return reduce(hcat, [Vector(f(u .+ bump(i), q, 0.0) .- f(u .- bump(i), q, 0.0)) / 2h for i in eachindex(u)])
end

"""The shortest period `p` with which the last `2p` entries of `s` repeat, and the durations of the last two cycles."""
function find_cycle(s, d)
    n = length(s)
    for p in 1:n÷3
        all(s[n-i] == s[n-i-p] for i in 0:2p-1) || continue
        return p, sum(d[n-p+1:n]), sum(d[n-2p+1:n-p])
    end
    return nothing
end

"""One noise-free run: `:stable`, `:saddle`, `:growing`, `:periodic` (with its speed ratio) or `:irregular`."""
function outcome(f, q, z, decode; T, dt)
    sol = solve(SDEProblem(f, zero_noise, z, (0.0, T), q; seed=UInt64(1)), EM(); dt, adaptive=false, saveat=T / 40_000,
                maxiters=typemax(Int))
    v = [norm(f(x, q, 0.0)) for x in sol.u]
    # ≤ rather than <, so a run that starts on an equilibrium (speed 0 throughout) counts as settled
    if v[end] <= 1e-7 * maximum(v)
        return count(>(1e-7), real.(eigvals(jac(f, q, sol.u[end])))) == 0 ? (:stable, NaN) : (:saddle, NaN)
    end
    s, st = Int[], Float64[]
    for (a, x) in zip(sol.t, sol.u)
        k = decode(x)
        k > 0 && (isempty(s) || last(s) != k) && (push!(s, k); push!(st, a))
    end
    length(s) < 8 && return (:irregular, NaN)
    d = diff(st)
    c = find_cycle(s[2:end-1], d[2:end])
    c === nothing && return (:irregular, NaN)
    _, last_cycle, before = c
    last_cycle / before > 1.2 && return (:growing, NaN)
    tail = v[end÷2:end]
    return (:periodic, maximum(tail) / minimum(tail))
end

"""
    classify(m, spec; Tnoisy=40sum(spec.τ), Tfree=150sum(spec.τ), starts=5, ghost_ratio=100) -> (; class, outcomes)
    classify(f, q, z0, decode; noise, Tnoisy, Tfree, dt, starts=5, ghost_ratio=100) -> (; class, outcomes)

The dynamical class of a model: `:heteroclinic`, `:excitable`, `:ghost`, `:limit_cycle` or
`:unknown`. A noisy run of length `Tnoisy` at the specification's noise gives `starts − 1` points
from its second half, plus the start itself; a noise-free run of length `Tfree` from each ends in
one of

- a stable equilibrium → excitable;
- a saddle, or cycles that keep growing (each over 1.2× the one before) → heteroclinic;
- a repeating cycle → ghost if its peak speed is at least `ghost_ratio` times its slowest, else limit
  cycle;
- anything else → unknown.

The class is unknown unless every start agrees. `outcomes` describes each start's run. The second
form classifies any vector field `f(u, q, t)` with a decoder `decode(u)` (values ≤ 0 are ignored),
stepped at `dt`.
"""
function classify(f, q, z0, decode; noise::Real, Tnoisy::Real, Tfree::Real, dt::Real, starts::Integer=5, ghost_ratio::Real=100.0)
    qn = merge(q, (; η=noise))
    sol = solve(SDEProblem(f, noise_eta, z0, (0.0, Tnoisy), qn; seed=UInt64(11)), EM(); dt, adaptive=false,
                saveat=Tnoisy / 2000, maxiters=typemax(Int))
    points = [[z0]; sol.u[round.(Int, range(length(sol.u) ÷ 2, length(sol.u); length=starts - 1))]]
    results = [outcome(f, qn, p, decode; T=Tfree, dt) for p in points]
    labels = map(results) do (o, r)
        o === :stable ? :excitable : o in (:saddle, :growing) ? :heteroclinic : o === :irregular ? :unknown :
            r >= ghost_ratio ? :ghost : :limit_cycle
    end
    class = allequal(labels) ? first(labels) : :unknown
    outcomes = join(("$(o)" * (isnan(r) ? "" : "($(round(r; sigdigits=3)))") for (o, r) in results), " ")
    return (; class, outcomes)
end

function classify(m::AbstractNetworkModel, spec::TransitionSpec; Tnoisy::Real=40sum(spec.τ), Tfree::Real=150sum(spec.τ),
                  dt::Real=0.02 / speed(m), kw...)
    q = parameters(m)
    return classify(drift_function(m), q, class_start(m), u -> class_state(m, q, u); noise=spec.noise, Tnoisy, Tfree, dt, kw...)
end
