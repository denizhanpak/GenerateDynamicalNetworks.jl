# Why the CTRNN fits fail: how each state's dwell responds to each cell's time constant, at the three
# seeds and at the three worm fits. The correction passes assume the response is +1 on the diagonal
# and 0 elsewhere. Run from notebooks/ with
#   julia --project=. -t 16 ../benchmarks/phase5/ctrnn_diagnose.jl
include(joinpath(@__DIR__, "backends.jl"))

_, P, τ = only(c for c in NAMED if c[1] == "worm ring")
spec = (; P, τ, form=:u, ε₀=0.1, sample=WORM_SPEC.sample)
const N = 4
const IDX = N^2+N+1:N^2+2N
const STEP = 0.3

function dwell_at(θ; T=250sum(spec.τ), seeds=1:32)
    m = try ctrnn_model(spec, θ) catch err; err isa ArgumentError ? nothing : rethrow() end
    m === nothing && return fill(NaN, N), NaN, NaN
    r = simulate_frames(m; T, seeds, spec.sample, dtscale=0.02)
    any(x -> x.failed, r) && return fill(NaN, N), NaN, m.α[1]
    s = score(r, spec)
    return s.dwell, s.off, m.α[1]
end

function report(name, θ)
    d0, off, rate = dwell_at(θ)
    @printf("\n%s\n  dwell %s (targets %s), off-graph %.3f, rate %.1f\n", name, round.(d0; sigdigits=3), round.(spec.τ; sigdigits=3), off, rate)
    @printf("  τ %s  g %.3g  W diag %s\n", round.(exp.(θ[IDX]); sigdigits=3), exp(θ[end]), round.([θ[(i-1)*N+i] for i in 1:N]; sigdigits=3))
    println("  d log dwell(row) / d log τ(column), from a +", STEP, " step in log τ:")
    for i in 1:N
        row = map(1:N) do k
            θp = copy(θ)
            θp[IDX[k]] += STEP
            return log(dwell_at(θp)[1][i] / d0[i]) / STEP
        end
        println("    ", join((@sprintf("%7.2f", x) for x in row), " "))
    end
    flush(stdout)
end

for r in (:excitable, :oscillator, :fold)
    report("seed $r", ctrnn_seed(spec, r))
end
for r in (:excitable, :oscillator, :fold)
    report("fit $r", vec(readdlm(joinpath(@__DIR__, "results", "theta_ctrnn_$(r)_worm_ring_0.1.csv"), ',')))
end
