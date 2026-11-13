# Specification, example networks and the measurement rules: no simulation.

@testset "spec_problems (the Phase 1 cases)" begin
    @test isempty(spec_problems(ring(4), ones(4), 1e-5))
    @test spec_problems([0 0.5 0.4; 0 0 1; 1 0 0], ones(3), 1e-5) == ["rows of P must sum to 1"]
    @test spec_problems(ring(3), [1, 0, 1.0], 1e-5) == ["every dwell must be positive"]
    @test spec_problems(ring(4), ones(3), 1e-5) == ["P has 4 states but τ has 3"]
    @test spec_problems(ring(3), ones(3), 0.0) == ["noise must be positive: noise-free dwell times are not fixed"]
    @test spec_problems([0 1 0; 0 0 1; 0 0 1.0], ones(3), 1e-5) == ["every state must be reachable from every other"]
    @test isempty(spec_problems([0.5 0.5 0; 0 0 1; 1 0 0], ones(3), 1e-5))      # a self-loop is a valid spec
    @test spec_problems(zeros(2, 3), ones(2), 0.1) == ["P must be square, not (2, 3)"]
    @test "P has a negative entry" in spec_problems([0 1.5 -0.5; 0 0 1; 1 0 0], ones(3), 0.1)
    @test "a network needs at least two states" in spec_problems(ones(1, 1), [1.0], 0.1)
end

@testset "TransitionSpec" begin
    s = TransitionSpec(ring(3), [1.0, 3.0, 10.0]; noise=0.1)
    @test nstates(s) == 3 && s.sample == 0.1 && s.noise == 0.1 && s.names == ["state 1", "state 2", "state 3"]
    @test edge_list(s) == [(1, 2), (2, 3), (3, 1)] && successors(s, 3) == [1]
    @test sprint(show, s) == "TransitionSpec(3 states, 3 edges, noise 0.1, sample 0.1)"
    @test_throws ArgumentError TransitionSpec(ring(3), [1.0, 0.0, 1.0]; noise=0.1)
    @test_throws ArgumentError TransitionSpec(ring(3), ones(3); noise=0.1, sample=0.0)
    @test_throws ArgumentError TransitionSpec(ring(3), ones(3); noise=0.1, names=["a", "b"])
    err = try TransitionSpec([0 0.5 0.4; 0 0 1; 1 0 0], [1.0, 0.0, 1.0]; noise=0.0) catch e; e end
    @test occursin("rows of P must sum to 1", err.msg) && occursin("dwell", err.msg) && occursin("noise", err.msg)
    g = SimpleDiGraph(3); add_edge!(g, 1, 2); add_edge!(g, 1, 3); add_edge!(g, 2, 1); add_edge!(g, 3, 1)
    @test TransitionSpec(g, ones(3); noise=0.1).P == [0 0.5 0.5; 1 0 0; 1 0 0]
    @test successors([0 0.5 0.5; 1 0 0; 1 0 0], 1) == [2, 3]
end

@testset "examples" begin
    @test length(example_names()) == 10 && allunique(example_names())
    for name in example_names()
        s = example(name; noise=1e-3)
        @test s isa TransitionSpec && s.noise == 1e-3
        @test s.sample ≈ (startswith(name, "worm") ? 1 / 29.97 : minimum(s.τ) / 10)
    end
    @test example("foraging").P[2, :] == [0, 0, 0.7, 0.3, 0]
    @test example("Δ-clique"; sample=0.5).sample == 0.5
    @test_throws ArgumentError example("no such graph")
    @test ring(3) == [0 1 0; 0 0 1; 1 0 0]
    for seed in 1:20
        P = random_network(Xoshiro(seed), 6, 0.35)
        @test isempty(spec_problems(P, ones(6), 0.1))
        @test all(i -> P[i, i] == 0 && count(>(0), P[i, :]) <= 2, 1:6)
        @test all(p -> p == 0 || p == 1 || 0.15 <= p <= 0.85, P)
    end
    @test random_network(Xoshiro(7), 5, 0.4) == random_network(Xoshiro(7), 5, 0.4)
end

@testset "measure, verdict, off_graph, score" begin
    # The first visit of a run is dropped (it starts part-way through) and the last has no end.
    runs = [(state=[1, 2, 1, 2, 1], t=[0.0, 1.0, 3.0, 4.0, 6.0], failed=false)]
    m = measure(runs, 2)
    @test m.counts == [0 1; 2 0] && m.samples == [[1.0], [2.0, 2.0]] && m.dwell == [1.0, 2.0]
    @test isnan(m.sem[1]) && m.sem[2] == 0.0
    flip = [0 1; 1 0.0]
    @test verdict(m, flip, [1.0, 2.0]).pass
    v = verdict(m, flip, [1.2, 2.0])
    @test !v.pass && v.err ≈ [1 / 1.2 - 1, 0] && isempty(v.branch)
    @test score(runs, (; P=flip, τ=[1.0, 2.0])).f == 0
    @test score([(state=[1, 2], t=[0.0, 1.0], failed=true)], (; P=flip, τ=[1.0, 2.0])).f == 1000

    hub = [0 0.5 0.5; 1 0 0; 1 0 0]
    even = (; counts=[0 50 50; 100 0 0; 0 0 0], dwell=ones(3))
    @test verdict(even, hub, ones(3)).pass
    skew = (; counts=[0 70 30; 100 0 0; 0 0 0], dwell=ones(3))
    v = verdict(skew, hub, ones(3))
    @test !v.pass && [b.ok for b in v.branch] == [false, false]
    @test v.branch[1] == (; from=1, to=2, designed=0.5, measured=0.7, ok=false)
    stray = (; counts=[0 50 50; 99 0 1; 0 0 0], dwell=ones(3))
    @test off_graph(stray, hub) ≈ 1 / 200 && verdict(stray, hub, ones(3)).pass
    @test !verdict((; counts=[0 50 50; 97 0 3; 0 0 0], dwell=ones(3)), hub, ones(3)).pass
end
