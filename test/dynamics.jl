# The simulator and the classifier on small models written here against the backend interface.

import GenerateDynamicalNetworks: drift_function, parameters, initial_state, speed, frame_state, project, frame_steps

# A double well: du = a(u − u³) dt + ε dW. State 1 is u > 0.5, state 2 is u < −0.5; between them the last
# state holds. With `floor`, every step is projected onto u ≥ floor.
struct DoubleWell <: AbstractNetworkModel
    a::Float64
    ε::Float64
    floor::Float64
end
DoubleWell(a, ε) = DoubleWell(a, ε, -Inf)
well(u, q, t) = q.a .* (u .- u .^ 3)
drift_function(::DoubleWell) = well
parameters(m::DoubleWell) = (; a=m.a, ε=m.ε)
initial_state(::DoubleWell) = SVector(1.0)
speed(m::DoubleWell) = m.a
frame_state(::DoubleWell, q, u) = u[1] > 0.5 ? 1 : u[1] < -0.5 ? 2 : 0
project(m::DoubleWell, u) = isfinite(m.floor) ? max.(u, m.floor) : nothing

# du = u² dt blows up at t = 1 from u = 1.
struct Blowup <: AbstractNetworkModel end
blow(u, q, t) = u .^ 2
drift_function(::Blowup) = blow
parameters(::Blowup) = (; ε=1e-6)
initial_state(::Blowup) = SVector(1.0)
speed(::Blowup) = 1.0
frame_state(::Blowup, q, u) = 1

@testset "simulate" begin
    m = DoubleWell(1.0, 0.7)
    @test frame_steps(m, 0.1, 0.01) == (10, 0.01) && frame_steps(m, 0.1, 0.03) == (4, 0.025)
    runs = simulate(m; T=2000.0, seeds=1:8, sample=0.1)
    @test length(runs) == 8 && !any(r -> r.failed, runs)
    @test all(r -> all(in((1, 2)), r.state) && all(!=(0), diff(r.state)) && issorted(r.t), runs)
    @test first(runs[1].state) == 1                                  # it starts in the right-hand well
    # a run depends only on its seed
    again = simulate(m; T=2000.0, seeds=reverse(1:8), sample=0.1)
    @test all(k -> again[9-k].state == runs[k].state && again[9-k].t == runs[k].t, 1:8)
    spec = TransitionSpec([0 1; 1 0], [1.0, 1.0]; noise=0.7, sample=0.1)
    @test simulate(m, spec; T=2000.0, seeds=1:8)[3].t == runs[3].t
    # the two wells are mirror images, so their mean dwells agree within the sampling error
    meas = measure(runs, 2)
    @test sum(meas.counts) > 1000 && abs(meas.dwell[1] - meas.dwell[2]) < 4hypot(meas.sem...)
    # projection: a floor above −0.5 keeps the run out of state 2
    @test all(r -> r.state == [1], simulate(DoubleWell(1.0, 0.7, -0.4); T=200.0, seeds=1:4, sample=0.1))
    # a run that leaves the bounds is stopped and marked failed
    bad = simulate(Blowup(); T=5.0, seeds=1:2, sample=0.1)
    @test all(r -> r.failed && last(r.t) < 1.0, bad)
end

@testset "validate and accept" begin
    m = DoubleWell(1.0, 0.7)
    spec = TransitionSpec([0 1; 1 0], [12.0, 12.0]; noise=0.7, sample=0.1)
    c = validate(m, spec; seeds=1:8, T=2000.0)
    meas = measure(simulate(m, spec; T=2000.0, seeds=1:8), 2)
    @test c.dwell == meas.dwell && c.counts == meas.counts && c.failed == 0
    @test c.err ≈ meas.dwell ./ 12 .- 1 && c.rms ≈ sqrt(mean(log.(meas.dwell ./ 12) .^ 2))
    @test c.pass == (all(abs.(c.err) .<= 0.05) && c.off < 0.01)
    @test occursin(c.pass ? "PASS" : "fail", sprint(show, c))
    a = accept(m, spec; T=400.0, set1=1:4, set2=5:8)
    @test a.set1.dwell == validate(m, spec; seeds=1:4, T=400.0).dwell
    @test a.half_step.dwell == validate(m, spec; seeds=1:4, T=400.0, dtscale=0.005).dwell
    @test a.pass == (a.set1.pass && a.set2.pass && a.half_step.pass)
    @test startswith(sprint(show, a), "Acceptance: ")
    blown = validate(Blowup(), spec; seeds=1:2, T=5.0)
    @test blown.failed == 2 && !blown.pass
end

# Vector fields of known class, from the prototype's classifier fixtures (research/benchmarks/phase5/classes.jl).
theta3(z, q, t) = SVector(z[1] * (1 - z[1]^2 - z[2]^2) - z[2] * (1 - z[1] + q.r), z[2] * (1 - z[1]^2 - z[2]^2) + z[1] * (1 - z[1] + q.r))
ctrnn(y, q, t) = (-y .+ q.W * (1 ./ (1 .+ exp.(-q.g .* (y .+ q.b))))) ./ q.τ
lv_u(u, q, t) = u .* (q.α .- q.ρ * (u .* u))

@testset "classify" begin
    @test classify(DoubleWell(1.0, 0.7), TransitionSpec([0 1; 1 0], [12.0, 12.0]; noise=0.7); Tnoisy=50.0, Tfree=200.0).class === :excitable
    theta(r) = classify(theta3, (; r), SVector(0.0, 1.0), z -> z[1] > 0.5 ? 1 : 2; noise=1e-4, Tnoisy=400.0, Tfree=2000.0, dt=0.005)
    @test theta(1e-3).class === :ghost
    @test theta(1.0).class === :limit_cycle
    beer = (; W=SMatrix{2,2}(4.5, -1.0, 1.0, 4.5), b=SVector(-2.75, -1.75), τ=SVector(0.273, 0.273), g=1.0)
    quadrant(b) = y -> (on = y .+ b .> 0; on[1] ? (on[2] ? 4 : 1) : (on[2] ? 3 : 2))
    @test classify(ctrnn, beer, SVector(log(9.0), -log(9.0)) .- beer.b, quadrant(beer.b); noise=1e-3, Tnoisy=20.0, Tfree=100.0,
                   dt=0.001).class === :limit_cycle
    # Horchler's Lotka–Volterra 3-ring (u form) with saddle values ν = 2: a stable heteroclinic cycle
    ρ = SMatrix{3,3}([i == j ? 1.0 : mod1(j + 1, 3) == i ? 0.5 : 2.0 for i in 1:3, j in 1:3])
    r = classify(lv_u, (; α=SVector(1.0, 1.0, 1.0), ρ), SVector(0.95, 1e-3, 1e-3), u -> argmax(abs.(u)); noise=1e-3,
                 Tnoisy=100.0, Tfree=1500.0, dt=0.02)
    @test r.class === :heteroclinic
    @test length(split(r.outcomes)) == 5
end
