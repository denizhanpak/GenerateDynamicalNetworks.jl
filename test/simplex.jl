# The simplex backend: the model from θ, its saddles, the fit's start, a short fit.

@testset "model" begin
    spec = example("foraging"; noise=0.1)
    n, edges = nstates(spec), edge_list(spec)
    ne = length(edges)
    θ = [log(3.0); log.(range(0.5, 1.5; length=ne)); log.(range(0.8, 1.2; length=n))]
    m = GDN.simplex_model(spec, θ)
    @test m isa LVModel && m.form === :u && m.visible == n && m.α ≈ fill(3.0, n) && allequal(m.α) && m.ε == fill(0.1, n)
    @test_throws ArgumentError GDN.simplex_model(spec, θ[1:end-1])
    # at vertex i it leaves at s·aᵢⱼ towards each successor j, contracts at s·cᵢ along the others and 2s along its own
    q = GDN.parameters(m)
    e, c = exp.(θ[2:1+ne]), exp.(θ[2+ne:end])
    for i in 1:n
        u = SVector{n}(ntuple(j -> j == i ? 1.0 : 0.0, n))
        @test maximum(abs, GDN.drift_function(m)(u, q, 0.0)) < 1e-12
        want = [j == i ? -6.0 : spec.P[i, j] > 0 ? 3e[findfirst(==((i, j)), edges)] : -3c[i] for j in 1:n]
        @test sort(real.(GDN.eigvals(GDN.jac(GDN.drift_function(m), q, u)))) ≈ sort(want) rtol = 1e-6
    end
    st = GDN.simplex_start(spec)
    @test st.seeded && length(st.θ0) == 1 + ne + n && all(st.lower .<= st.θ0 .<= st.upper)
    runs = simulate(GDN.simplex_model(spec, st.θ0), spec; T=20sum(spec.τ), seeds=1:4)
    @test !any(r -> r.failed, runs) && all(r -> length(r.state) > 10 && all(in(1:n), r.state), runs)
end

@testset "fit" begin
    @test_throws ArgumentError Simplex(; passes=0)
    @test Simplex().seconds == 300 && Simplex().T === nothing
    spec = TransitionSpec(ring(3), [1.0, 2.0, 3.0]; noise=0.1)
    b = Simplex(; passes=1, runs=4, T=30sum(spec.τ), budget=12, popsize=6, seconds=60)
    m = fit(spec, b)
    @test m isa LVModel && m.visible == 3
    @test fit(spec, b).ρ == m.ρ                         # a fit depends on the seeds, not on time or threads
    runs = simulate(m, spec; T=60.0, seeds=1:4)
    @test !any(r -> r.failed, runs) && all(r -> all(in(1:3), r.state), runs)
end
