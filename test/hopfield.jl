# The Hopfield backend.

@testset "vertex design" begin
    spec = example("foraging"; noise=0.1)
    L = GDN.hop_layout(spec)
    rough, c = GDN.rough_rates(spec)
    m, worst = GDN.hop_model(spec, L, rough; fast=c, contract=c)
    @test worst < 1e-6
    # the same rows every time (OSQP's step-size interval is fixed, not timed)
    d = GDN.hop_design(L, rough; fast=c, contract=c)
    @test d.J == m.J && d.b == m.b && d.k == m.k
    q, Λ = GDN.parameters(m), GDN.hop_rates(m)
    # every vertex is an equilibrium whose Jacobian has eigenvalues equal to the designed rates
    for v in 1:2^m.n
        ψ = SVector{m.n}(ntuple(j -> GDN.corner_bit(v, j) == 1 ? Float64(π) : 0.0, m.n))
        @test maximum(abs, GDN.hop_drift(ψ, q, 0.0)) < 1e-12
        @test sort(real.(GDN.eigvals(GDN.jac(GDN.hop_drift, q, ψ)))) ≈ sort(Λ[:, v]) atol = 1e-6
        @test GDN.frame_state(m, q, ψ) == m.node_of[v]
    end
    # departures as designed, and every other direction at a used vertex contracting by at least the margin
    steps = GDN.path_steps(L)
    for (v, j, e, s) in steps
        @test Λ[j, v] ≈ (s == 1 ? rough[e] : c) rtol = 1e-6
    end
    departing = Set((v, j) for (v, j, _, _) in steps)
    for v in unique(reduce(vcat, L.paths)), j in 1:m.n
        (v, j) in departing || @test Λ[j, v] <= -c / 4 + 1e-6
    end
end

@testset "layouts" begin
    spec = example("flip-flop")                         # one two-way pair
    L = GDN.onehot_layout(spec.P)
    rel = GDN.relayed_edges(spec.P)
    @test L.M == nstates(spec) + length(rel) && L.corner_of == [1 << (i - 1) + 1 for i in 1:nstates(spec)]
    for (p, e) in zip(L.paths, edge_list(spec))
        @test length(p) == (e in rel ? 5 : 3)
        @test all(k -> count_ones((p[k] - 1) ⊻ (p[k+1] - 1)) == 1, 1:length(p)-1)
    end
    for name in ("foraging", "flip-flop")
        spec = example(name)
        L = GDN.hop_layout(spec)
        rough, c = GDN.rough_rates(spec)
        @test GDN.hop_design(L, rough; fast=c, contract=c).worst < 1e-3
    end
end

@testset "fit, fallback, deadline and resume" begin
    spec = TransitionSpec(ring(3), [0.1, 0.3, 1.0]; noise=0.1, sample=0.01)
    L = GDN.hop_layout(spec)
    _, c = GDN.rough_rates(spec)
    out0 = GDN.hop_seed(spec, L)
    @test length(out0) == 3 && all(isfinite, out0)
    # failure injection: the k-th model build has no design
    calls, poison = Ref(0), Ref(0)
    function build(out)
        calls[] += 1
        calls[] == poison[] && throw(GDN.NoDesign())
        return first(GDN.hop_model(spec, L, out; fast=c, contract=c))
    end
    function attempt(k)
        calls[], poison[] = 0, k
        file = tempname()
        try
            return GDN.keep_last_design(build, file) do
                r = GDN.correct_rates(spec, build, out0; passes=1, cap=2c, T=30sum(spec.τ), checkpoint=file)
                build(r.out)
            end
        finally
            rm(file; force=true)
        end
    end
    # the seed itself has no design and nothing is saved yet: a clear error
    @test_throws ArgumentError attempt(1)
    # one pass runs, then the final update has none: keep that pass's rates, with a warning
    m = @test_logs (:warn, r"no pairwise design for the rates after pass 1") attempt(2)
    @test m isa HopModel && all(isfinite, m.J)
    @test attempt(0).J != m.J                           # the normal path returns the update
    b = Hopfield(; passes=1)
    f = fit(spec, b)
    @test f isa HopModel && !any(r -> r.failed, simulate(f, spec; T=20.0, seeds=1:4))
    file = tempname()
    @test_throws OutOfTime fit(spec, b; checkpoint=file, deadline=time())
    @test fit(spec, b; checkpoint=file).J == f.J
    rm(file)
    @test HopModel(spec, out0).n == L.M
end
