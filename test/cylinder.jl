# The cylinder backend: the model from θ and its options, its saddles, the decoder, a short fit.

@testset "model" begin
    spec = example("foraging"; noise=0.1)
    n, edges = nstates(spec), edge_list(spec)
    ne = length(edges)
    θ = [log(3.0); log.(range(0.5, 1.5; length=ne))]
    m = CylinderModel(spec, θ)
    @test m.s ≈ 3.0 && m.visible == n && m.ε == 0.1 && (m.k, m.Lω, m.pwindow, m.ywindow) == (2.0, 1.5625, 0.25, 0.5)
    @test_throws ArgumentError CylinderModel(spec, θ[1:end-1])
    @test_throws ArgumentError CylinderModel(spec, θ; k=0)
    @test_throws ArgumentError CylinderModel(spec, θ; Lω=-1)
    @test_throws ArgumentError CylinderModel(spec, θ; pwindow=0.6)
    @test_throws ArgumentError CylinderModel(spec, θ; ywindow=0)
    @test GDN.initial_state(m) == SVector{ne + 1}([fill(0.1, ne); 1.0]) && GDN.class_start(m) == SVector{ne + 1}([fill(1e-3, ne); 1.0])
    # at state i (p = i, y = 0) it leaves at s·eᵢⱼ along each edge out of i and contracts along everything else,
    # whatever k and Lω are
    e = exp.(θ[2:end])
    for (k, Lω) in ((2.0, 1.5625), (3.0, 1.0))
        mk = CylinderModel(spec, θ; k, Lω)
        q = GDN.parameters(mk)
        for i in 1:n
            z = SVector{ne + 1}([zeros(ne); i])
            @test maximum(abs, GDN.drift_function(mk)(z, q, 0.0)) < 1e-12
            λ = real.(GDN.eigvals(GDN.jac(GDN.drift_function(mk), q, z)))
            @test sort(filter(>(0), λ)) ≈ sort([3e[a] for a in eachindex(edges) if edges[a][1] == i]) rtol = 1e-6
        end
    end
    # the decoder: state j only near p = j with every edge cell small, otherwise 0 (the last state holds)
    q = GDN.parameters(m)
    y(v) = fill(v, ne)
    @test GDN.frame_state(m, q, SVector{ne + 1}([y(0.1); 2.1])) == 2
    @test GDN.frame_state(m, q, SVector{ne + 1}([y(0.1); 2.3])) == 0
    @test GDN.frame_state(m, q, SVector{ne + 1}([y(0.6); 2.0])) == 0
    @test GDN.frame_state(m, q, SVector{ne + 1}([y(0.1); 0.1])) == 0
    @test GDN.frame_state(m, q, SVector{ne + 1}([y(0.1); 1e30])) == 0
    @test GDN.frame_state(m, q, SVector{ne + 1}([y(0.1); NaN])) == 0
    wide = CylinderModel(spec, θ; pwindow=0.4, ywindow=0.7)
    @test GDN.frame_state(wide, GDN.parameters(wide), SVector{ne + 1}([y(0.6); 2.3])) == 2
    st = GDN.cylinder_start(spec, 1.5625)
    @test st.seeded && length(st.θ0) == 1 + ne && all(st.lower .<= st.θ0 .<= st.upper)
    runs = simulate(CylinderModel(spec, st.θ0), spec; T=20sum(spec.τ), seeds=1:4)
    @test !any(r -> r.failed, runs) && all(r -> length(r.state) > 10 && all(in(1:n), r.state), runs)
end

@testset "fit" begin
    @test_throws ArgumentError Cylinder(; passes=0)
    @test_throws ArgumentError Cylinder(; pwindow=0)
    b = Cylinder(; k=2.5)
    @test b.seconds == 300 && b.k == 2.5 && b.Lω == 1.5625 && b.T === nothing
    spec = TransitionSpec(ring(3), [1.0, 2.0, 3.0]; noise=0.1)
    b = Cylinder(; passes=1, runs=4, T=30sum(spec.τ), budget=12, popsize=6, seconds=60)
    m = fit(spec, b)
    @test m isa CylinderModel && m.visible == 3
    @test fit(spec, b).Lα == m.Lα                       # a fit depends on the seeds, not on time or threads
    runs = simulate(m, spec; T=60.0, seeds=1:4)
    @test !any(r -> r.failed, runs) && all(r -> all(in(1:3), r.state), runs)
end
