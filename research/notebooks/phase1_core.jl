### A Pluto.jl notebook ###
# v1.1.0

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 8aecac42-da3a-11f1-845c-6571baae0394
begin
	import Pkg
	Pkg.activate(@__DIR__)   # notebooks/Project.toml, which finds SHCTools next to this package
	Pkg.instantiate()
	using SHCTools, Graphs, Plots, PlutoUI
	using Distributions, LinearAlgebra, Random, Roots, Statistics
end

# ╔═╡ 8b12546a-da3a-11f1-8ac4-cbdd012a65f9
md"""
# GenerateDynamicalNetworks: Phase 1 experiments

Phase 1 is the layer every backend will share: the specification, the graph checks, the decoder, the measurements and the backend interface. This notebook tries those ideas out before any of them become package code.

Each section asks one question, because each answer fixes a design decision:

| Section | Question |
|:--------|:---------|
| 1 | What should a specification reject? |
| 2 | How many graphs do the backends' rules exclude, and can hidden states bring them back? |
| 3 | Does the choice of decoder change the measured dwell? |
| 4 | Can the measurements be trusted, and how much simulation do they need? |
| 5 | Does one interface fit very different backends? |
| 6 | What this settles |

Everything here is a prototype. The Lotka-Volterra model comes from SHCTools, and nothing in this notebook has been promoted to the package.
"""

# ╔═╡ 8b125690-da3a-11f1-9736-bdbe994f1301
TableOfContents(title="Contents", depth=2)

# ╔═╡ 8b1256cc-da3a-11f1-9f35-f928a659b194
begin
	const state_colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
	const muted = "#52514e"
	default(; fontfamily="sans-serif", linewidth=2, framestyle=:axes, grid=true, gridalpha=0.12,
		tickfontcolor=muted, guidefontcolor=muted, foreground_color_axis=muted,
		foreground_color_border=muted, legend_foreground_color=:transparent,
		legend_background_color=:transparent, size=(720, 320), left_margin=3Plots.mm,
		bottom_margin=3Plots.mm, palette=state_colors)
	sig(x, d=3) = x isa Number && isfinite(x) ? round(x; sigdigits=d) : x
	pct(x, d=1) = isfinite(x) ? string(d == 0 ? round(Int, 100x) : round(100x; digits=d), "%") : "n/a"
	# A Markdown table from a header and rows of values.
	mdtable(header, rows) = Markdown.parse(join(
		["| " * join(r, " | ") * " |" for r in [header, fill("---", length(header)), rows...]], "\n"))
end

# ╔═╡ 8b125708-da3a-11f1-a5e7-b3abda18aea9
md"""
## 1. What should a specification reject?

A specification is the transition matrix `P`, the mean dwell vector `τ` and a noise level. `P[i, j]` is the probability that state `j` follows state `i`, and a dwell runs from entering a state to entering the next one.

Some problems are wrong for every backend, so they belong here. Others, such as a 2-cycle, only matter to particular constructions, so they belong to the backend (section 2). A self-loop is the clearest case: a CTRNN might not care, so the specification allows it and the backend decides.
"""

# ╔═╡ 8b12573a-da3a-11f1-b0a6-db0717df6e64
function spec_problems(P, τ, noise)
	size(P, 1) == size(P, 2) || return ["P must be square, not $(size(P))"]
	n = size(P, 1)
	problems = String[]
	n >= 2 || push!(problems, "a network needs at least two states")
	length(τ) == n || push!(problems, "P has $n states but τ has $(length(τ))")
	all(>=(0), P) || push!(problems, "P has a negative entry")
	sum(P; dims=2) ≈ ones(n) || push!(problems, "rows of P must sum to 1")
	all(>(0), τ) || push!(problems, "every dwell must be positive")
	noise > 0 || push!(problems, "noise must be positive: noise-free dwell times are not fixed")
	is_strongly_connected(SimpleDiGraph(P .> 0)) ||
		push!(problems, "every state must be reachable from every other")
	return problems
end

# ╔═╡ 8b125780-da3a-11f1-83fc-1ffb7a12e0f5
begin
	struct NetworkSpec
		P::Matrix{Float64}
		τ::Vector{Float64}
		noise::Float64
		names::Vector{String}
	end

	function NetworkSpec(P::AbstractMatrix, τ; noise=1e-5, names=["state $i" for i in eachindex(τ)])
		problems = spec_problems(P, τ, noise)
		isempty(problems) || throw(ArgumentError(join(problems, "; ")))
		return NetworkSpec(float(P), float(τ), noise, collect(string.(names)))
	end

	# A graph alone means every branch is equally likely.
	NetworkSpec(g::SimpleDiGraph, τ; kwargs...) =
		NetworkSpec(let A = float(Matrix(adjacency_matrix(g))); A ./ max.(sum(A; dims=2), 1) end, τ; kwargs...)
end

# ╔═╡ 8b1257a8-da3a-11f1-be35-d33c00e30456
ring(n) = [mod1(i + 1, n) == j ? 1.0 : 0.0 for i in 1:n, j in 1:n]

# ╔═╡ 8b1257da-da3a-11f1-a81b-0be8ce3bf40a
spec_cases = [
	("a valid ring", ring(4), ones(4), 1e-5),
	("a row that doesn't sum to 1", [0 .5 .4; 0 0 1; 1 0 0.], ones(3), 1e-5),
	("a dwell of 0", ring(3), [1, 0, 1.0], 1e-5),
	("τ too short", ring(4), ones(3), 1e-5),
	("zero noise", ring(3), ones(3), 0.0),
	("an absorbing state", [0 1 0; 0 0 1; 0 0 1.], ones(3), 1e-5),
	("a self-loop", [.5 .5 0; 0 0 1; 1 0 0.], ones(3), 1e-5),
]

# ╔═╡ 8b12580c-da3a-11f1-bbcf-b9de9ef0c4be
mdtable(["specification", "verdict"], [[name, let p = spec_problems(P, τ, e)
	isempty(p) ? "accepted" : join(p, "; ") end] for (name, P, τ, e) in spec_cases])

# ╔═╡ 8b125834-da3a-11f1-aa1f-858b5c3e94ab
md"""
Zero noise is rejected on purpose. In a noise-free attracting cycle each pass takes longer than the last, so a dwell time isn't a fixed number there. The dwell vector is a statement about the noisy system.

## 2. How many graphs do the rules exclude?

The constructions disagree about what they can build, so this asks how much each rule actually costs on random strongly connected graphs. Each rule below is a structural check. No model is simulated in this section.

| Rule | Comes from | What goes wrong |
|:-----|:-----------|:----------------|
| No self-loops | every saddle construction | a saddle can't connect to itself |
| No 2-cycles | simplex, Horchler | both nodes settle into a stable mix |
| No Δ-cliques | Horchler | the shortcut starves the longer path |
| Branch noise solvable | Horchler | each node has one noise value, so its branches must agree |

The last rule is new here. Your notebook's `branch_noise` assigns noise ratios one branch at a time and gives up when two branches share a successor. But each two-way branch is really one linear equation in ``\log\varepsilon``:

```math
\log\varepsilon_j - \log\varepsilon_k = \log\tan\frac{\pi p_j}{2}
```

So all the branches together are a linear system, and the honest question is whether it has a solution.
"""

# ╔═╡ 8b125870-da3a-11f1-a0b4-a3269f267403
begin
	self_loops(P) = [i for i in axes(P, 1) if P[i, i] > 0]
	two_cycles(P) = [(i, j) for i in axes(P, 1) for j in axes(P, 1) if i < j && P[i, j] > 0 && P[j, i] > 0]
	delta_cliques(P) = let A = P .> 0, n = axes(A, 1)
		[(a, b, c) for a in n for b in n for c in n if allunique((a, b, c)) && A[a, b] && A[a, c] && A[b, c]]
	end
end

# ╔═╡ 8b125898-da3a-11f1-99bc-e93535ba21ea
"""
How far the branches of `P` are from sharing one noise value per node: 0 if they can, `Inf` if a
branch wider than two is uneven, which has no formula yet.
"""
function noise_residual(P)
	n = size(P, 1)
	rows, b = Vector{Float64}[], Float64[]
	equation!(j, k, c) = (r = zeros(n); r[j] = 1; r[k] = -1; push!(rows, r); push!(b, c))
	for i in 1:n
		s = findall(>(0), P[i, :])
		if length(s) == 2
			equation!(s[1], s[2], log(tan(π * P[i, s[1]] / 2)))
		elseif length(s) > 2
			allequal(P[i, s]) || return Inf
			for k in s[2:end]
				equation!(s[1], k, 0.0)
			end
		end
	end
	isempty(rows) && return 0.0
	A = reduce(vcat, r' for r in rows)
	return norm(A * (A \ b) - b)
end

# ╔═╡ 8b1258d4-da3a-11f1-ba7e-7d14a6d7173d
"""`branch_noise` from the SHCTools notebook, reduced to whether it would accept `P`."""
function greedy_accepts(P)
	n = size(P, 1)
	ε = ones(n)
	for i in 1:n
		s = findall(>(0), P[i, :])
		length(s) == 2 || continue
		ratio = tan(π * P[i, s[1]] / 2)
		ε[s[1]], ε[s[2]] = sqrt(ratio), 1 / sqrt(ratio)
	end
	for i in 1:n
		s = findall(>(0), P[i, :])
		length(s) > 2 && (!allequal(P[i, s]) || !allequal(ε[s])) && return false
		length(s) == 2 && !(2 / π * atan(ε[s[1]] / ε[s[2]]) ≈ P[i, s[1]]) && return false
	end
	return true
end

# ╔═╡ 8b125906-da3a-11f1-94cd-3daab19bea9f
md"""
### Hidden states as a general repair

Your notebook inserts a hidden state on one edge of a Δ-clique. The same move should cover two more rules. Putting a hidden state on one edge of a 2-cycle turns it into a 3-cycle. Putting one on a shared successor's edge gives that branch its own noise value, because the branch now races against the hidden state instead of the shared one.

`repair` applies all three until nothing is left to fix. A wide uneven branch and a self-loop are out of its reach.
"""

# ╔═╡ 8b125938-da3a-11f1-b23d-030e2959e2ce
function subdivide(P, a, b)
	h, n = size(P, 1) + 1, size(P, 1)
	Q = [P zeros(n); zeros(h)']
	Q[a, h], Q[h, b], Q[a, b] = Q[a, b], 1, 0
	return Q
end

# ╔═╡ 8b12596a-da3a-11f1-938d-956fd4d22552
function repair(P)
	Q, visible = float(P), size(P, 1)
	while true
		if !isempty(two_cycles(Q))
			a, b = first(two_cycles(Q))
			Q = subdivide(Q, a, b)
		elseif !isempty(delta_cliques(Q))
			_, b, c = first(delta_cliques(Q))
			Q = subdivide(Q, b, c)
		elseif isfinite(noise_residual(Q)) && noise_residual(Q) > 1e-9
			# Give the first branch that shares a successor with another its own hidden state.
			branching = [i for i in axes(Q, 1) if count(>(0), Q[i, :]) >= 2]
			shared = [(i, j) for i in branching for j in findall(>(0), Q[i, :])
				if any(k != i && Q[k, j] > 0 for k in branching)]
			isempty(shared) && return (; Q, hidden=size(Q, 1) - visible, ok=false)
			Q = subdivide(Q, shared[1]...)
		else
			return (; Q, hidden=size(Q, 1) - visible, ok=isempty(self_loops(Q)) && isfinite(noise_residual(Q)))
		end
	end
end

# ╔═╡ 8b125992-da3a-11f1-9cdd-6f1db0dcc263
"""
A random strongly connected graph on `n` states with no self-loops, edge density `d`. Branches
are equal, or uneven and two-way at most when `uneven`, so that every rule stays meaningful.
"""
function random_spec_matrix(rng, n, d; uneven=false)
	while true
		A = [i != j && rand(rng) < d for i in 1:n, j in 1:n]
		if uneven
			for i in 1:n
				s = findall(A[i, :])
				length(s) > 2 && (A[i, shuffle(rng, s)[3:end]] .= false)
			end
		end
		is_strongly_connected(SimpleDiGraph(A)) || continue
		P = float(A)
		for i in 1:n
			s = findall(A[i, :])
			if uneven && length(s) == 2
				p = rand(rng, Uniform(0.15, 0.85))
				P[i, s] = [p, 1 - p]
			else
				P[i, s] .= 1 / length(s)
			end
		end
		return P
	end
end

# ╔═╡ 8b1259ea-da3a-11f1-ab48-cf7603d7921f
md"""
Choose the number of states. Each point is 400 random graphs, with every branch two-way and uneven.
"""

# ╔═╡ 8b125a14-da3a-11f1-a8e8-b158e6fe320b
@bind n_rules Slider(5:8; default=6, show_value=true)

# ╔═╡ 8b125a46-da3a-11f1-a828-21bb255ebf9f
rule_rates = let densities = 0.15:0.05:0.6, rng = Xoshiro(1), trials = 400
	map(densities) do d
		graphs = [random_spec_matrix(rng, n_rules, d; uneven=true) for _ in 1:trials]
		repaired = repair.(graphs)
		(; d,
			no_two=mean(isempty(two_cycles(P)) for P in graphs),
			no_two_delta=mean(isempty(two_cycles(P)) && isempty(delta_cliques(P)) for P in graphs),
			greedy=mean(isempty(two_cycles(P)) && isempty(delta_cliques(P)) && greedy_accepts(P) for P in graphs),
			linear=mean(isempty(two_cycles(P)) && isempty(delta_cliques(P)) && noise_residual(P) < 1e-9 for P in graphs),
			repaired=mean(r.ok for r in repaired),
			hidden=mean(r.hidden for r in repaired))
	end
end

# ╔═╡ 8b125a78-da3a-11f1-9e60-d97edf839bd6
fig_rules = let x = getfield.(rule_rates, :d)
	series = [("no 2-cycles", :no_two, state_colors[1]),
		("no 2-cycles or Δ-cliques", :no_two_delta, state_colors[2]),
		("all rules, linear noise solve", :linear, state_colors[3]),
		("all rules, greedy noise", :greedy, state_colors[4]),
		("after repair", :repaired, state_colors[5])]
	p = plot(; xlabel="edge density", ylabel="fraction of random graphs accepted", ylims=(0, 1.05),
		legend=:outerright, size=(760, 340))
	for (label, key, color) in series
		plot!(p, x, getfield.(rule_rates, key); label, color, marker=:circle, markersize=3,
			markerstrokewidth=0)
	end
	p
end

# ╔═╡ 8b125aaa-da3a-11f1-872a-f33a3da2681f
mdtable(["density", "no 2-cycles", "+ no Δ-cliques", "+ greedy noise", "+ linear noise", "after repair", "hidden states"],
	[[sig(r.d, 2), pct(r.no_two, 0), pct(r.no_two_delta, 0), pct(r.greedy, 0), pct(r.linear, 0), pct(r.repaired, 0),
		sig(r.hidden, 2)] for r in rule_rates])

# ╔═╡ 8b125ad2-da3a-11f1-b8ab-edc540a61a21
md"""
## 3. Does the decoder change the dwell?

A decoder turns activity into a list of visits, so it defines what a dwell *is*. Your code uses two definitions. The SHCTools notebook counts a state as entered when its activity crosses the edge of its neighbourhood, and the state then holds until another one crosses. `FittedModel.jl` takes the largest variable at every sample.

The test case is the worm ring, because its 26:1 dwell ratio is the hardest case so far. All decoders read the same trajectory, so any difference between them comes from the decoder and not from sampling.
"""

# ╔═╡ 8b125b04-da3a-11f1-9e6f-9723c38ad97f
worm = (names=["Forward", "Pause 1", "Reverse", "Pause 2"], τ=[5.329, 0.441, 1.945, 0.208])

# ╔═╡ 8b125b2c-da3a-11f1-a793-4bbc33d51a3d
"""One shared α, with ν setting each dwell, as `dwell_rates` in the SHCTools notebook."""
function ring_model(τ; ε, κ=0.1, νmin=2.0)
	n, β = length(τ), ones(length(τ))
	αᵢ(i, ν) = first(lv_params(τ[i], ε, β[i], ν; κ))
	α = maximum(αᵢ(i, νmin) for i in 1:n)
	ν = [find_zero(v -> αᵢ(i, v) - α, (νmin, 1e6)) for i in 1:n]
	return (; τ, β, κ, ε=fill(float(ε), n), α=fill(α, n), ν, ρ=createcycle(fill(α, n), β, ν))
end

# ╔═╡ 8b125b5e-da3a-11f1-a322-cdb012b960f5
function lv_run(m; T, seed=1)
	n = length(m.α)
	dt = 0.01 / maximum(m.α)
	x0 = fill(minimum(m.ε), n)
	x0[1] = (1 - m.κ) * m.β[1]
	every = max(1, round(Int, minimum(m.τ) / 200 / dt))
	sol = lv_integrate(0.0:dt:T, x0, m.ρ, m.α, m.ε; every, seed)
	return (; t=sol.t, X=Array(sol))
end

# ╔═╡ 8b125b86-da3a-11f1-8e5e-194d781a6bd7
begin
	"""Enter a state when its activity crosses `edge`, and hold it until another state crosses."""
	function latch(X, t; edge=0.9)
		state, entered = Int[], Float64[]
		for k in axes(X, 2)
			v, i = findmax(view(X, :, k))
			if v > edge && (isempty(state) || last(state) != i)
				push!(state, i)
				push!(entered, t[k])
			end
		end
		return (; state, t=entered)
	end

	"""The largest variable at each sample, as in `FittedModel.jl`."""
	function dominant(X, t)
		state, entered = Int[], Float64[]
		for k in axes(X, 2)
			i = argmax(view(X, :, k))
			if isempty(state) || last(state) != i
				push!(state, i)
				push!(entered, t[k])
			end
		end
		return (; state, t=entered)
	end
end

# ╔═╡ 8b125bb8-da3a-11f1-af9f-6d12441f1735
"""
Dwell statistics from visits. Only complete dwells count: the last visit has no end, and the
first is dropped because the run starts part-way through it.
"""
function measure(visits, n)
	state, t = visits.state[2:end], visits.t[2:end]
	from, to, d = state[1:end-1], state[2:end], diff(t)
	samples = [d[from .== i] for i in 1:n]
	return (; samples, dwell=[isempty(s) ? NaN : mean(s) for s in samples],
		sem=[length(s) < 2 ? NaN : std(s) / sqrt(length(s)) for s in samples],
		counts=[count((from .== i) .& (to .== j)) for i in 1:n, j in 1:n])
end

# ╔═╡ 8b125bf6-da3a-11f1-809f-cf774777618b
off_graph(m, P) = sum(m.counts[c] for c in CartesianIndices(P) if P[c] == 0; init=0) / max(sum(m.counts), 1)

# ╔═╡ 8b125c1c-da3a-11f1-a5c7-6f04547ff98e
md"""
Before trusting `measure`, check it on hand-built visits whose answers can be read off directly.
"""

# ╔═╡ 8b125c44-da3a-11f1-858b-19498c0bec3f
hand_checks = let
	v = (state=[1, 2, 3, 1, 2, 3, 1], t=[0.0, 5.0, 6.0, 8.0, 12.0, 13.0, 16.0])
	m = measure(v, 3)
	short = measure((state=[1, 2], t=[0.0, 1.0]), 2)
	skip = measure((state=[1, 2, 1, 3, 1, 2], t=0.0:5.0), 3)
	flicker = latch([0.95 0.5 0.95; 0.0 0.5 0.0], 0:2)
	[("drops the first visit and the last", m.samples == [[4.0], [1.0, 1.0], [2.0, 3.0]]),
	 ("counts follow the sequence", m.counts == [0 1 0; 0 0 2; 2 0 0]),
	 ("too few visits gives NaN, not an error", all(isnan, short.dwell)),
	 ("an off-graph step is counted", off_graph(skip, ring(3)) > 0),
	 ("latch holds through a dip", flicker.state == [1])]
end

# ╔═╡ 8b125c80-da3a-11f1-b83f-0d8355e0e760
mdtable(["check", "result"], [[name, r === true ? "pass" : "FAIL: $r"] for (name, r) in hand_checks])

# ╔═╡ 8b125cb2-da3a-11f1-b288-adb6976300bc
decoder_runs = map((1e-5, 1e-3, 1e-2)) do ε
	m = ring_model(worm.τ; ε)
	lv_run(m; T=1.0)    # so that the timing below leaves out compilation
	seconds = @elapsed r = lv_run(m; T=800.0)
	(; ε, m, r, seconds)
end

# ╔═╡ 8b125cda-da3a-11f1-8b4b-4b35e88838bf
decoders = (("latch 0.9", (X, t) -> latch(X, t; edge=0.9)),
	("latch 0.5", (X, t) -> latch(X, t; edge=0.5)),
	("dominant", dominant))

# ╔═╡ 8b125d02-da3a-11f1-be7d-e302d1e68ae7
decoder_table = [let m = measure(f(run.r.X, run.r.t), 4)
	(; ε=run.ε, name, dwell=m.dwell, visits=length.(m.samples), off=off_graph(m, ring(4)), m)
end for run in decoder_runs for (name, f) in decoders]

# ╔═╡ 8b125d5e-da3a-11f1-9c07-4947276341dd
mdtable(["noise", "decoder", "Forward", "Pause 1", "Reverse", "Pause 2", "fewest visits", "off-graph"],
	[[r.ε, r.name, [pct(r.dwell[i] / worm.τ[i] - 1) for i in 1:4]..., minimum(r.visits), pct(r.off)]
	 for r in decoder_table])

# ╔═╡ 8b125d8c-da3a-11f1-ae1a-bfaa44add86d
md"""
Those are errors against the design target. The next chart moves the `latch` edge at one noise level, to see how much the threshold itself matters.
"""

# ╔═╡ 8b125dbe-da3a-11f1-9de4-659dd6e4029d
@bind noise_choice Select([1e-5 => "noise 1e-5", 1e-3 => "noise 1e-3", 1e-2 => "noise 1e-2"]; default=1e-3)

# ╔═╡ 8b125de8-da3a-11f1-af12-f5013d076f8c
edge_sweep = let run = decoder_runs[findfirst(r -> r.ε == noise_choice, decoder_runs)]
	edges = 0.3:0.05:0.95
	(; edges, errors=[measure(latch(run.r.X, run.r.t; edge=e), 4).dwell ./ worm.τ .- 1 for e in edges])
end

# ╔═╡ 8b125e24-da3a-11f1-aaa2-d5fc8c37fddf
fig_edges = let p = plot(; xlabel="latch edge", ylabel="dwell error (%)", legend=:outerright,
		xlims=(0.28, 1.08), size=(760, 320))
	hline!(p, [0]; color=muted, linewidth=1, label="")
	for i in 1:4
		y = 100 .* getindex.(edge_sweep.errors, i)
		plot!(p, edge_sweep.edges, y; label=worm.names[i],
			color=state_colors[i], marker=:circle, markersize=3, markerstrokewidth=0)
		annotate!(p, last(edge_sweep.edges) + 0.01, last(y), text(worm.names[i], 8, muted, :left))
	end
	p
end

# ╔═╡ 8b125e62-da3a-11f1-9b42-995691930736
md"""
The chart has a bend near 0.4, and two measurements explain the shape. The first is how far activity has fallen when two states hand over: if a handover happens below an edge, no state ever crosses it at that moment, so edges below that level all behave like `dominant`. The second is how long each state takes to rise from 0.5 to 0.9. A dwell runs from one entry to the next, so moving the edge shifts a state's dwell by the difference between its own rise time and its successor's.
"""

# ╔═╡ 8b125e94-da3a-11f1-9017-b5ca81c1ddbb
handover = map(decoder_runs) do run
	switches = dominant(run.r.X, run.r.t).t[2:end]
	level = median(maximum(view(run.r.X, :, searchsortedfirst(run.r.t, s))) for s in switches)
	low, high = latch(run.r.X, run.r.t; edge=0.5), latch(run.r.X, run.r.t; edge=0.9)
	rise = map(1:4) do i
		lags = [high.t[k] - last(low.t[(low.t .<= high.t[k]) .& (low.state .== i)])
			for k in eachindex(high.t) if high.state[k] == i && any((low.t .<= high.t[k]) .& (low.state .== i))]
		median(lags)
	end
	(; run.ε, level, rise)
end

# ╔═╡ 8b125ec6-da3a-11f1-9b37-01fdd979431d
mdtable(["noise", "activity at handover", ("rise 0.5 → 0.9: " .* worm.names)...],
	[[r.ε, sig(r.level, 2), sig.(r.rise, 2)...] for r in handover])

# ╔═╡ 8b125ef4-da3a-11f1-b2a3-916cdd880b5f
md"""
## 4. Can the measurements be trusted?

Three separate worries here, each checked against something with a known answer.

### Censoring

A run has a fixed length, so the dwell in progress at the end is thrown away, and long dwells are the most likely to be cut. That should bias the mean low when runs are short, which matters because optimisation wants runs as short as possible. A semi-Markov process has exactly known dwell times, so it measures this bias directly.
"""

# ╔═╡ 8b125f32-da3a-11f1-93fb-07eadd3b1c8d
function semimarkov_visits(rng, P, τ; T, cv, start=1)
	state, entered = Int[], Float64[]
	s, t = start, 0.0
	while t < T
		push!(state, s)
		push!(entered, t)
		t += rand(rng, Gamma(1 / cv^2, τ[s] * cv^2))
		s = rand(rng, Categorical(P[s, :]))
	end
	return (; state, t=entered)
end

# ╔═╡ 8b125f64-da3a-11f1-99f0-55f210a68f2a
censoring = let rng = Xoshiro(2), P = ring(4), cycle = sum(worm.τ), reps = 1500
	map(Iterators.product((0.1, 0.5, 1.0), (2, 3, 5, 10, 20, 50))) do (cv, cycles)
		one_run = [measure(semimarkov_visits(rng, P, worm.τ; T=cycles * cycle, cv), 4) for _ in 1:reps]
		means = filter(isfinite, getindex.(getfield.(one_run, :dwell), 1))
		pooled = reduce(vcat, getindex.(getfield.(one_run, :samples), 1))
		(; cv, cycles, averaged=mean(means) / worm.τ[1] - 1, pooled=mean(pooled) / worm.τ[1] - 1)
	end
end

# ╔═╡ 8b125f96-da3a-11f1-9e8d-696f28085270
fig_censoring = let p = plot(; xlabel="run length (cycles)", ylabel="bias in longest state's dwell (%)",
		xscale=:log10, xticks=([2, 5, 10, 20, 50], ["2", "5", "10", "20", "50"]), xlims=(1.8, 55), ylims=(-52, 3),
		legend=:outerright, size=(760, 320))
	hline!(p, [0]; color=muted, linewidth=1, label="")
	for (j, cv) in enumerate((0.1, 0.5, 1.0))
		r = censoring[j, :]
		y = 100 .* getfield.(r, :pooled)
		plot!(p, getfield.(r, :cycles), y; label="dwell CV $cv",
			color=state_colors[j], marker=:circle, markersize=4, markerstrokewidth=0)
		annotate!(p, 2.1, first(y) - 3, text("CV $cv", 8, muted, :left))
	end
	p
end

# ╔═╡ 8b125fca-da3a-11f1-bd99-8568e1852ab4
mdtable(["dwell CV", "cycles", "bias, pooled dwells", "bias, mean of run means"],
	[[r.cv, r.cycles, pct(r.pooled), pct(r.averaged)] for r in vec(censoring)])

# ╔═╡ 8b125ffc-da3a-11f1-8c47-31270182a436
md"""
### Serial correlation

A standard error of `std/√n` assumes dwells are independent. Manicom et al. (2024) show that heteroclinic networks can carry memory from one visit to the next, in which case those error bars would be too narrow. This measures the correlation between each dwell and the next one on the real Lotka-Volterra runs.
"""

# ╔═╡ 8b126036-da3a-11f1-8933-0f319a8b688c
serial = map(decoder_runs) do run
	v = latch(run.r.X, run.r.t)
	d = diff(v.t[2:end])
	s = v.state[2:end-1]
	lagged = [cor(d[s .== i][1:end-1], d[s .== i][2:end]) for i in 1:4]
	next = [let k = findall(==(i), s[1:end-1]); cor(d[k], d[k.+1]) end for i in 1:4]
	(; ε=run.ε, n=count(==(1), s), lagged, next)
end

# ╔═╡ 8b126068-da3a-11f1-b31b-c7eaffd186d9
mdtable(["noise", "visits per state", "chance level (2/√n)", "next dwell of same state", "next state's dwell"],
	[[r.ε, r.n, sig(2 / sqrt(r.n), 2), join(sig.(r.lagged, 2), ", "), join(sig.(r.next, 2), ", ")] for r in serial])

# ╔═╡ 8b12608e-da3a-11f1-8bed-5f691aaced61
md"""
### How much simulation is enough?

To resolve a mean within 5% at 95% confidence takes about ``(1.96\,\mathrm{CV}/0.05)^2`` visits. A branch probability ``p`` known to ``\pm0.03`` takes about ``p(1-p)(1.96/0.03)^2``. The dwell CV comes from the runs above, and the run time from how long those runs took, so this is a direct estimate of what one fitness evaluation costs.
"""

# ╔═╡ 8b1260c0-da3a-11f1-ab84-977fcbdab939
budget = map(decoder_runs) do run
	m = measure(latch(run.r.X, run.r.t), 4)
	cv = [std(s) / mean(s) for s in m.samples]
	dwell_visits = ceil(Int, maximum((1.96 .* cv ./ 0.05) .^ 2))
	branch_visits = ceil(Int, 0.5 * 0.5 * (1.96 / 0.03)^2)
	per_cycle = run.seconds / (800.0 / sum(worm.τ))
	(; run.ε, cv, dwell_visits, branch_visits, per_cycle,
		dwell_seconds=dwell_visits * per_cycle, branch_seconds=branch_visits * per_cycle)
end

# ╔═╡ 8b12613a-da3a-11f1-ad6a-e5ffaf5944f3
mdtable(["noise", "dwell CV by state", "visits for dwell", "visits for a 50/50 branch", "CPU s, dwell", "CPU s, branch"],
	[[r.ε, join(sig.(r.cv, 2), ", "), r.dwell_visits, r.branch_visits, sig(r.dwell_seconds, 2), sig(r.branch_seconds, 2)]
	 for r in budget])

# ╔═╡ 8b12616c-da3a-11f1-8c99-fd6d2e618b2b
md"""
## 5. Does one interface fit different backends?

The interface has to work for a Lotka-Volterra saddle, a CTRNN and a phase oscillator, and those describe "being in a state" very differently. The proposal is that a backend answers three questions and the core does the rest:

- **`supports(c, spec)`**: reasons this backend can't build the specification, empty if it can.
- **`build(c, spec)`**: the model.
- **`trajectory(c, model, T; seed)`**: times and an activation matrix `X`, states × samples, each entry from 0 to 1.

The activation matrix is the key decision. A decoder that only reads `X` never needs to know what the model is. To test that, three backends go through one pipeline, and the third is a stand-in for Phase 6: phases that visit cluster patterns, with a state's activation being how close the phases are to its pattern, ignoring the global phase.
"""

# ╔═╡ 8b12619e-da3a-11f1-bbee-399850987550
begin
	abstract type Construction end
	struct ToySemiMarkov <: Construction
		cv::Float64
	end
	struct LVRing <: Construction end
	struct ToyClusters <: Construction
		oscillators::Int
		cv::Float64
	end
end

# ╔═╡ 8b1261d0-da3a-11f1-ae5b-2368ea2fc83d
begin
	supports(::Construction, spec) = String[]
	function supports(::LVRing, spec)
		n = size(spec.P, 1)
		n >= 3 || return ["a ring needs at least three states"]
		spec.P == ring(n) ? String[] : ["this prototype only builds the ring 1 → 2 → … → n → 1"]
	end
end

# ╔═╡ 8b126202-da3a-11f1-bec3-079340977aae
begin
	build(c::ToySemiMarkov, spec) = (; backend=c, spec)
	build(::LVRing, spec) = (; backend=LVRing(), spec, lv=ring_model(spec.τ; ε=spec.noise))
	function build(c::ToyClusters, spec)
		rng = Xoshiro(5)
		patterns = [rand(rng, (0.0, π / 2, π, 3π / 2), c.oscillators) for _ in eachindex(spec.τ)]
		return (; backend=c, spec, patterns)
	end
end

# ╔═╡ 8b126228-da3a-11f1-ae7a-9d90a8736f45
begin
	sample_times(T, τ) = 0.0:minimum(τ)/50:T
	trajectory(m, T; seed=1) = trajectory(m.backend, m, T; seed)

	function trajectory(::ToySemiMarkov, m, T; seed)
		v = semimarkov_visits(Xoshiro(seed), m.spec.P, m.spec.τ; T, cv=m.backend.cv)
		t = sample_times(T, m.spec.τ)
		X = zeros(length(m.spec.τ), length(t))
		for (k, tₖ) in enumerate(t)
			X[v.state[searchsortedlast(v.t, tₖ)], k] = 1
		end
		return (; t, X)
	end

	trajectory(::LVRing, m, T; seed) = lv_run(m.lv; T, seed)

	function trajectory(::ToyClusters, m, T; seed)
		rng = Xoshiro(seed)
		v = semimarkov_visits(rng, m.spec.P, m.spec.τ; T, cv=m.backend.cv)
		t = sample_times(T, m.spec.τ)
		X = zeros(length(m.spec.τ), length(t))
		z = zeros(ComplexF64, m.backend.oscillators)
		handoff = minimum(m.spec.τ) / 5    # phases take this long to slide between patterns
		for (k, tₖ) in enumerate(t)
			j = searchsortedlast(v.t, tₖ)
			target = cis.(m.patterns[v.state[j]])
			previous = j == 1 ? target : cis.(m.patterns[v.state[j-1]])
			w = clamp((tₖ - v.t[j]) / handoff, 0, 1)
			z .= ((1 - w) .* previous .+ w .* target) .* cis.(0.05 .* randn(rng, length(z)) .+ 3tₖ)
			θ = angle.(z)
			for i in axes(X, 1)
				X[i, k] = abs(mean(cis.(θ .- m.patterns[i])))    # ignores the global phase
			end
		end
		return (; t, X)
	end
end

# ╔═╡ 8b12625c-da3a-11f1-916b-7185ba91d1c1
function generate_stub(spec, c; T=800.0, seed=1)
	reasons = supports(c, spec)
	isempty(reasons) || return (; status=:unsupported, reasons)
	tr = trajectory(build(c, spec), T; seed)
	m = measure(latch(tr.X, tr.t; edge=0.5), length(spec.τ))
	return (; status=:measured, reasons, dwell=m.dwell, error=m.dwell ./ spec.τ .- 1,
		off=off_graph(m, spec.P), visits=minimum(length.(m.samples)))
end

# ╔═╡ 8b12628e-da3a-11f1-a720-27d834e6ad39
interface_runs = let ringspec = NetworkSpec(ring(4), [2.0, 1.0, 0.5, 1.5])
	branch = NetworkSpec([0 1 0 0 0; 0 0 .7 .3 0; 0 0 0 0 1; 0 0 0 0 1; 1 0 0 0 0.], [2.0, 1.0, 0.5, 1.5, 1.0])
	[(name, label, generate_stub(s, c)) for (label, s) in (("ring", ringspec), ("branch", branch))
		for (name, c) in (("ToySemiMarkov", ToySemiMarkov(0.1)), ("LVRing", LVRing()), ("ToyClusters", ToyClusters(6, 0.1)))]
end

# ╔═╡ 8b1262b6-da3a-11f1-919b-3b97c0f1e0a0
mdtable(["backend", "spec", "result", "rms dwell error", "off-graph", "fewest visits"],
	[[name, label, r.status === :unsupported ? "refused: $(only(r.reasons))" : "measured",
		r.status === :measured ? pct(sqrt(mean(abs2, r.error))) : "",
		r.status === :measured ? pct(r.off) : "", r.status === :measured ? r.visits : ""]
	 for (name, label, r) in interface_runs])

# ╔═╡ 8b1262e8-da3a-11f1-a70c-a3679f91a5b2
findings = let
	r15 = first(rule_rates)
	row(ε, name) = only(filter(r -> r.ε == ε && r.name == name, decoder_table))
	spread(ε, i) = let e = [100 * (row(ε, n).dwell[i] / worm.τ[i] - 1) for n in ("latch 0.9", "latch 0.5", "dominant")]
		maximum(e) - minimum(e) end
	cell(cv, cycles) = only(filter(c -> c.cv == cv && c.cycles == cycles, vec(censoring)))
	lv = only(r for (name, label, r) in interface_runs if name == "LVRing" && label == "ring")
	clusters = only(r for (name, label, r) in interface_runs if name == "ToyClusters" && label == "ring")
	branch_cost = budget[2].branch_seconds
	dwell_cost = budget[2].dwell_seconds
	budget_text = "A 50/50 branch needs $(budget[1].branch_visits) visits against $(budget[1].dwell_visits) to " *
		"$(budget[end].dwell_visits) for dwell. At noise 1e-3 that is $(sig(branch_cost, 2)) CPU s against $(sig(dwell_cost, 2))"
	(; r15, spread, cell, row, lv, clusters, budget_text, serial_max=maximum(maximum(abs.([r.lagged; r.next])) * sqrt(r.n) / 2 for r in serial))
end

# ╔═╡ 8b12631a-da3a-11f1-98d7-6f03084ee8dd
md"""
## 6. What this settles

These are the design decisions for the package, at this notebook's default settings. Every number here is computed in the sections above.

| Topic | Result | Decision |
|:------|:-------|:---------|
| Specification | Each bad input gets its own reason | Keep `NetworkSpec` and `spec_problems`. Self-loops stay a backend question |
| Graph rules | At density 0.15 on $(n_rules) states, $(pct(findings.r15.no_two_delta)) of graphs pass unrepaired, falling to almost none by 0.35 | Repair belongs in the core, not in one backend |
| Hidden states | Repair handles every test graph (all branches two-way), using $(sig(findings.r15.hidden, 2)) to $(sig(rule_rates[end].hidden, 2)) hidden states | Promote `repair`, keeping a record of which states are hidden |
| Noise solve | Linear solve accepts $(pct(findings.r15.linear, 1)) against greedy's $(pct(findings.r15.greedy, 1)) at density 0.15 | Use the linear solve. The gain is small, but it's also the simpler rule |
| Decoder | At noise 1e-2 the decoders differ by $(sig(findings.spread(1e-2, 4), 2)) percentage points on Pause 2 but only $(sig(findings.spread(1e-2, 1), 2)) on Forward | Default to `latch(edge=0.5)`: it lost no visits, unlike 0.9 |
| Censoring | At CV 0.5 and 5 cycles, pooling is biased by $(pct(findings.cell(0.5, 5).pooled)) and averaging run means by $(pct(findings.cell(0.5, 5).averaged)) | Average per-run means, with at least 5 cycles per run |
| Serial correlation | Largest correlation is $(sig(findings.serial_max, 2)) × the chance level, with no consistent sign | `std/√n` is fine on rings. Retest on branches |
| Budget | $(findings.budget_text) | Branch precision sets the cost of a fitness evaluation |
| Interface | LV and the cluster stand-in ran through one pipeline at $(pct(sqrt(mean(abs2, findings.lv.error)))) and $(pct(sqrt(mean(abs2, findings.clusters.error)))) rms | Keep `supports`, `build`, `trajectory` and the activation matrix |

### What this doesn't show

- **Repair is structural only.** It says a repaired graph passes the rules, not that it simulates correctly. A hidden state also needs dwell time from somewhere. Phase 2 has to test this, especially for 2-cycles and shared successors, where repair is new.
- **The race formula hasn't been tested on chained branches.** The linear solve assumes the two-way formula still holds when branches share successors. Phase 2 needs a simulation for that too.
- **The decoder is not the main error.** Forward's dwell error is $(pct(findings.row(1e-2, "latch 0.5").dwell[1] / worm.τ[1] - 1, 0)) at noise 1e-2 under every decoder. That comes from the design, so Phases 2 and 3 have to deal with it.
- **Only one real backend ran.** The cluster backend is a stand-in that visits patterns by construction, and no CTRNN has been through the interface yet.
- **The correlation test is weak.** With about 100–200 visits per state it can only see correlations above roughly 0.2, and memory at branches is where Manicom et al. predict it.
- **Phase 0 hasn't happened.** These timings use StochasticDiffEq on one thread, and the optimiser and simulator benchmarks are still to come.
"""

# ╔═╡ Cell order:
# ╠═8aecac42-da3a-11f1-845c-6571baae0394
# ╟─8b12546a-da3a-11f1-8ac4-cbdd012a65f9
# ╟─8b125690-da3a-11f1-9736-bdbe994f1301
# ╟─8b1256cc-da3a-11f1-9f35-f928a659b194
# ╟─8b125708-da3a-11f1-a5e7-b3abda18aea9
# ╠═8b12573a-da3a-11f1-b0a6-db0717df6e64
# ╠═8b125780-da3a-11f1-83fc-1ffb7a12e0f5
# ╟─8b1257a8-da3a-11f1-be35-d33c00e30456
# ╟─8b1257da-da3a-11f1-a81b-0be8ce3bf40a
# ╟─8b12580c-da3a-11f1-bbcf-b9de9ef0c4be
# ╟─8b125834-da3a-11f1-aa1f-858b5c3e94ab
# ╠═8b125870-da3a-11f1-a0b4-a3269f267403
# ╠═8b125898-da3a-11f1-99bc-e93535ba21ea
# ╠═8b1258d4-da3a-11f1-ba7e-7d14a6d7173d
# ╟─8b125906-da3a-11f1-94cd-3daab19bea9f
# ╠═8b125938-da3a-11f1-b23d-030e2959e2ce
# ╠═8b12596a-da3a-11f1-938d-956fd4d22552
# ╠═8b125992-da3a-11f1-9cdd-6f1db0dcc263
# ╟─8b1259ea-da3a-11f1-ab48-cf7603d7921f
# ╠═8b125a14-da3a-11f1-a8e8-b158e6fe320b
# ╠═8b125a46-da3a-11f1-a828-21bb255ebf9f
# ╟─8b125a78-da3a-11f1-9e60-d97edf839bd6
# ╟─8b125aaa-da3a-11f1-872a-f33a3da2681f
# ╟─8b125ad2-da3a-11f1-b8ab-edc540a61a21
# ╠═8b125b04-da3a-11f1-9e6f-9723c38ad97f
# ╠═8b125b2c-da3a-11f1-a793-4bbc33d51a3d
# ╠═8b125b5e-da3a-11f1-a322-cdb012b960f5
# ╠═8b125b86-da3a-11f1-8e5e-194d781a6bd7
# ╠═8b125bb8-da3a-11f1-af9f-6d12441f1735
# ╠═8b125bf6-da3a-11f1-809f-cf774777618b
# ╟─8b125c1c-da3a-11f1-a5c7-6f04547ff98e
# ╟─8b125c44-da3a-11f1-858b-19498c0bec3f
# ╟─8b125c80-da3a-11f1-b83f-0d8355e0e760
# ╠═8b125cb2-da3a-11f1-b288-adb6976300bc
# ╠═8b125cda-da3a-11f1-8b4b-4b35e88838bf
# ╟─8b125d02-da3a-11f1-be7d-e302d1e68ae7
# ╟─8b125d5e-da3a-11f1-9c07-4947276341dd
# ╟─8b125d8c-da3a-11f1-ae1a-bfaa44add86d
# ╠═8b125dbe-da3a-11f1-9de4-659dd6e4029d
# ╠═8b125de8-da3a-11f1-af12-f5013d076f8c
# ╟─8b125e24-da3a-11f1-aaa2-d5fc8c37fddf
# ╟─8b125e62-da3a-11f1-9b42-995691930736
# ╠═8b125e94-da3a-11f1-9017-b5ca81c1ddbb
# ╟─8b125ec6-da3a-11f1-9b37-01fdd979431d
# ╟─8b125ef4-da3a-11f1-b2a3-916cdd880b5f
# ╠═8b125f32-da3a-11f1-93fb-07eadd3b1c8d
# ╠═8b125f64-da3a-11f1-99f0-55f210a68f2a
# ╟─8b125f96-da3a-11f1-9e8d-696f28085270
# ╟─8b125fca-da3a-11f1-bd99-8568e1852ab4
# ╟─8b125ffc-da3a-11f1-8c47-31270182a436
# ╠═8b126036-da3a-11f1-8933-0f319a8b688c
# ╟─8b126068-da3a-11f1-b31b-c7eaffd186d9
# ╟─8b12608e-da3a-11f1-8bed-5f691aaced61
# ╠═8b1260c0-da3a-11f1-ab84-977fcbdab939
# ╟─8b12613a-da3a-11f1-ad6a-e5ffaf5944f3
# ╟─8b12616c-da3a-11f1-8c99-fd6d2e618b2b
# ╠═8b12619e-da3a-11f1-bbee-399850987550
# ╠═8b1261d0-da3a-11f1-ae5b-2368ea2fc83d
# ╠═8b126202-da3a-11f1-bec3-079340977aae
# ╠═8b126228-da3a-11f1-ae7a-9d90a8736f45
# ╠═8b12625c-da3a-11f1-916b-7185ba91d1c1
# ╟─8b12628e-da3a-11f1-a720-27d834e6ad39
# ╟─8b1262b6-da3a-11f1-919b-3b97c0f1e0a0
# ╟─8b1262e8-da3a-11f1-a70c-a3679f91a5b2
# ╟─8b12631a-da3a-11f1-98d7-6f03084ee8dd
