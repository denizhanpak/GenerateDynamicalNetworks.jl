# What are the worm fits, dynamically? Three measurements, none of which changes a model:
#   1. every equilibrium of the noise-free field and its eigenvalues;
#   2. noise-free runs in log coordinates, read as a bout sequence (growing bouts = heteroclinic,
#      repeating bouts = limit cycle, a last bout that never ends = equilibrium or a very slow passage);
#   3. dwell against noise, compared with the Stone-Holmes formula the design uses.
# Run from notebooks/ with
#   julia --project=. -t 8 ../benchmarks/phase3/worm_class.jl
include(joinpath(@__DIR__, "worm.jl"))
using DelimitedFiles

const OUT = joinpath(@__DIR__, "results")
const NOISES = [0.1, 0.03, 0.01, 1e-3, 1e-4]

function load_fit(name)
    θ = vec(readdlm(joinpath(OUT, name), ','))
    α, ν = exp.(θ[1:4]), exp.(θ[5:8])
    return (; form=:u, P=WORM_P, visible=4, κ=0.1, β=ones(4), α, ν, ε=fill(0.1, 4), ρ=graph_rho(WORM_P, α, ones(4), ν))
end

const FORMULA = design(WORM_P, WORM_SPEC.τ; form=:u, ε₀=0.1)
const MODELS = ["user fit" => USER_FIT, "formula seed" => FORMULA, "search 1" => load_fit("worm_raw_polished_theta.csv"),
                "search 4" => load_fit("worm_repeat_4_theta.csv"), "search 2" => load_fit("worm_repeat_2_theta.csv")]

# ── 1. Equilibria ───────────────────────────────────────────────────────────────────────────

function equilibria(m)
    n = length(m.α)
    found = NamedTuple[]
    for bits in 1:(2^n - 1)
        S = [i for i in 1:n if bits >> (i - 1) & 1 == 1]
        F = lu(m.ρ[S, S]; check=false)
        if !issuccess(F)    # no isolated equilibrium on this face: none at all, or a whole line of them
            println("  singular block on ", S)
            continue
        end
        x = F \ m.α[S]
        all(>(1e-9), x) || continue
        full = zeros(n)
        full[S] = x
        u = sqrt.(full)
        growth = m.α .- m.ρ * full
        J = [(i == j ? growth[i] : 0.0) - 2u[i] * m.ρ[i, j] * u[j] for i in 1:n, j in 1:n]
        e = eigvals(J)
        push!(found, (; S, x, unstable=count(>(1e-9), real.(e)), maxre=maximum(real.(e)), maxim=maximum(abs.(imag.(e)))))
    end
    return found
end

# ── 2. Noise-free bouts, in y = log u so that nothing underflows ────────────────────────────

drift_y(y, q, t) = q.α .- q.ρ * exp.(2 .* y)
no_noise(y, q, t) = zero(y)

function bouts(m; T, y0)
    S = SVector{4}
    q = (; α=S(m.α), ρ=SMatrix{4,4}(m.ρ))
    prob = SDEProblem(drift_y, no_noise, S(fill(log(y0), 4) .* (1 .- [1, 0, 0, 0]) .+ log(sqrt(0.9)) .* [1, 0, 0, 0]), (0.0, T), q)
    sol = solve(prob, EM(); dt=0.01 / maximum(m.α), saveat=0.02, maxiters=typemax(Int), seed=UInt64(1))
    state, start = Int[], Float64[]
    for (y, t) in zip(sol.u, sol.t)
        i = argmax(y)
        (isempty(state) || last(state) != i) && (push!(state, i); push!(start, t))
    end
    return state, diff([start; T])
end

# ── 3. Dwell against noise ──────────────────────────────────────────────────────────────────

function sweep(name, m, io)
    rows = []
    for ε in NOISES
        t = @elapsed runs = simulate_frames(merge(m, (; ε=fill(ε, 4))); T=2000.0, seeds=9_200_001:9_200_032, sample=WORM_SPEC.sample, dtscale=0.01)
        meas = measure(runs, 4)
        failed = count(r -> r.failed, runs)
        off = off_graph(meas, WORM_P)
        for k in 1:4
            theory = try designed_dwell(:u, m.α[k], m.ν[k], ε, m.β[k], m.κ) catch; NaN end
            s = meas.samples[k]
            cv = length(s) > 2 ? std(s) / mean(s) : NaN
            push!(rows, (; ε, k, dwell=meas.dwell[k], theory, cv, n=length(s), off, failed))
            println(io, join([name, ε, k, meas.dwell[k], theory, cv, length(s), off, failed, m.α[k], m.ν[k], round(t; digits=1)], ","))
        end
        flush(io)
        @printf("  ε %-7g %5.1f s  failed %d  off %.3f  dwell %s  CV %s\n", ε, t, failed, off,
                join((@sprintf("%.3g", r.dwell) for r in rows[end-3:end]), " "), join((@sprintf("%.2f", r.cv) for r in rows[end-3:end]), " "))
        flush(stdout)
    end
    return rows
end

"""Least-squares slope of dwell on ln(1/ε) over the three smallest noises, divided by the formula's ν/α."""
function slope_ratio(rows, m, k)
    r = filter(x -> x.k == k && x.ε <= 0.01, rows)
    X = [ones(length(r)) log.(1 ./ getproperty.(r, :ε))]
    return (X \ getproperty.(r, :dwell))[2] / (m.ν[k] / m.α[k])
end

open(joinpath(OUT, "worm_class_sweep.csv"), "w") do io
    println(io, "model,noise,state,dwell,theory,cv,n,off,failed,alpha,nu,seconds")
    for (name, m) in MODELS
        println("── ", name)
        println("  α ", round.(m.α; sigdigits=4), "  ν ", round.(m.ν; sigdigits=4), "  Πν ", round(prod(m.ν); sigdigits=4),
                "  e-folds per dwell at target τα/ν ", round.(WORM_SPEC.τ .* m.α ./ m.ν; sigdigits=3))
        for e in equilibria(m)
            println("  equilibrium on ", e.S, ": unstable ", e.unstable, "  max Re ", round(e.maxre; sigdigits=3), "  max |Im| ", round(e.maxim; sigdigits=3))
        end
        for y0 in (1e-3, 1e-6)
            s, d = bouts(m; T=2000.0, y0)
            println("  noise-free from others at $y0: states ", first(s, 14), "  durations ", round.(first(d, 14); sigdigits=3), length(s) > 14 ? "  … $(length(s)) bouts" : "")
        end
        flush(stdout)
        rows = sweep(name, m, io)
        at = filter(r -> r.ε == 0.1, rows)
        println("  at noise 0.1, measured/formula: ", [round(r.dwell / r.theory; sigdigits=3) for r in at])
        println("  slope in ln(1/ε) over formula ν/α, ε ≤ 0.01: ", [round(slope_ratio(rows, m, k); sigdigits=3) for k in 1:4])
        flush(stdout)
    end
end
