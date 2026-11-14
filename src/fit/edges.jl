# Seed and correction for backends with one global speed and one rate per edge (two-population, simplex,
# cylinder): the prototype's Phase 4 rules.

"""
    edge_seed_rates(spec, edges) -> (; α, logs, seeded)

A shared speed `α` and a log rate per edge. Speeds and per-state rates come from the Horchler dwell
design with one noise value ([`dwell_rates`](@ref) in the u form); branches split by the low-noise
scaling, a rate ratio of about p^(1/L) with L = max(log(√0.1/ε), 1). `seeded` is false when the
design has no solution, in which case a rough guess from the dwells is used instead.
"""
function edge_seed_rates(spec, edges)
    n = size(spec.P, 1)
    L = max(log(sqrt(0.1) / spec.noise), 1.0)
    ε_eff = [effective_noise(fill(float(spec.noise), length(successors(spec.P, i)))) for i in 1:n]
    α, base, seeded = try
        a, ν = dwell_rates(:u, spec.τ, ε_eff, ones(n))
        a[1], -log.(ν), true
    catch err
        err isa ArgumentError || rethrow()
        s = 10 / minimum(spec.τ)
        s, log.(L ./ (s .* spec.τ)), false
    end
    logs = [base[i] + (log(spec.P[i, j]) - mean(log.(spec.P[i, successors(spec.P, i)]))) / L for (i, j) in edges]
    return (; α, logs, seeded)
end

"""
    correct_edges(build, θ0, spec, edges, lower, upper; passes, runs, T, dtscale=0.01, base=100_000) -> (trace, θ)

Correction passes on `θ = [log speed; log edge rates; …]`: every edge out of state i moves by
log(measured dwell / τᵢ), and the edges of a branch also by the branch error divided by L (the
sensitivity from the low-noise scaling). Returns one row per pass and the θ after the last update,
clamped to `[lower, upper]`.
"""
function correct_edges(build, θ0, spec, edges, lower, upper; passes, runs, T, dtscale=0.01, base=100_000)
    n = size(spec.P, 1)
    L = max(log(sqrt(0.1) / spec.noise), 1.0)
    θ = copy(θ0)
    trace = []
    for pass in 1:passes
        r = simulate(build(θ); T, seeds=base+1000pass+1:base+1000pass+runs, sample=spec.sample, dtscale)
        s = score(r, spec)
        push!(trace, (; pass, θ=copy(θ), f=s.f, dwell=s.dwell, off=s.off))
        all(isfinite, s.dwell) || break
        C = measure(r, n).counts
        change = zeros(length(edges))
        for (k, (i, j)) in enumerate(edges)
            out = successors(spec.P, i)
            error_of(b) = log(spec.P[i, b]) - log(max(C[i, b], 1) / max(sum(C[i, out]), 1))
            change[k] = log(s.dwell[i] / spec.τ[i]) + (error_of(j) - mean(error_of, out)) / L
        end
        θ[2:1+length(edges)] .+= change
        θ .= clamp.(θ, lower, upper)
    end
    return trace, θ
end

"""
    edge_rate_fit(build, spec, θ0, lower, upper; passes, runs, T, budget, popsize, seconds, label="", verbose=false) -> θ

The fit the simplex and cylinder backends share (the prototype's phase 4 `fit_backend`), on
`θ = [log speed; log edge rates; …]` in the box `[lower, upper]`: `passes` correction passes
([`correct_edges`](@ref)) from `θ0`. If the best of them passes a check on 32 held-out runs it is the
result; otherwise a CMA-ES search from it of at most `budget` evaluations or `seconds` seconds, then two
correction passes of 32 runs of length 4T.
"""
function edge_rate_fit(build, spec, θ0, lower, upper; passes, runs, T, budget, popsize, seconds, label="", verbose=false)
    edges = edge_list(spec.P)
    trace, _ = correct_edges(build, clamp.(θ0, lower, upper), spec, edges, lower, upper; passes, runs, T)
    θc = argmin(p -> p.f, trace).θ
    if validate(build(θc), spec; seeds=8_000_001:8_000_032, T=2T, dtscale=0.01).pass
        verbose && println("  $label: the correction passes pass the held-out check; no search")
        return θc
    end
    s = cma_search(build, spec, clamp.(θc, lower .+ 1e-6, upper .- 1e-6), 0.3, lower, upper; budget, popsize, runs, T,
                   dtscale=0.04, maxtime=seconds)
    verbose && println("  $label: search used $(s.evals) evaluations")
    _, θp = correct_edges(build, s.θ, spec, edges, lower, upper; passes=2, runs=32, T=4T, base=200_000)
    return θp
end
