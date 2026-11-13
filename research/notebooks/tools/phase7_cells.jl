begin
	import Pkg
	Pkg.activate(@__DIR__)   # notebooks/Project.toml
	Pkg.instantiate()
	using Plots, PlutoUI, DelimitedFiles, Printf
	# The packages the included files load. Pluto only re-imports what it sees in a cell, so they
	# are listed here too, or a cell rerun after a control moves can't find `solve` or `remake`.
	using SHCTools, StochasticDiffEq, StaticArrays, Graphs, Distributions, QuadGK, Roots, JuMP, OSQP
	using LinearAlgebra, Random, Statistics, Serialization
	import CMAEvolutionStrategy as CES
	const phase7 = joinpath(@__DIR__, "..", "benchmarks", "phase7")
	include(joinpath(phase7, "hopfield.jl"))      # the backend, with Phases 2 to 6 through it
	include(joinpath(phase7, "..", "phase5", "twoway_cases.jl"))   # the worm graph measured from the tracks
end
#---
md"""
# GenerateDynamicalNetworks: Phase 7, a Hopfield network on the cube

The optional phase. The plan: read Chossat & Krupa and implement their construction if it gives a usable algorithm.

Choices you made for this phase (2026-11-11):

| Choice | Setting |
|:--|:--|
| Construction | a general coupling matrix, designed one neuron (one row) at a time |
| Noise | on the angle ``\psi_j`` with ``x_j = \cos\psi_j``, so every ``x_j`` stays in ``[-1, 1]`` |
| Solver | JuMP + OSQP for each neuron's row |
| Scope | as Phase 6: the 3-ring pilot at noise ``10^{-3}`` and ``0.1``, then the six fixture graphs at both noises and the real worm graph at ``0.1``, with the same three checks and the class |

The code is in `benchmarks/phase7/`: `hopfield.jl` (model, layouts, fit), `precheck.jl`, `pilot.jl` and `suite.jl`.
"""
#---
TableOfContents(title="Contents", depth=2)
#---
begin
	# Slots 1–6 of the validated categorical palette, in fixed order.
	const state_colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
	const muted = "#52514e"
	default(; fontfamily="sans-serif", linewidth=2, framestyle=:axes, grid=true, gridalpha=0.12,
		tickfontcolor=muted, guidefontcolor=muted, foreground_color_axis=muted,
		foreground_color_border=muted, legend_foreground_color=:transparent,
		legend_background_color=:transparent, size=(720, 320), left_margin=3Plots.mm,
		bottom_margin=3Plots.mm)
	pct(x, d=1) = x isa Number && isfinite(x) ? "$(round(100x; digits=d))%" : "–"
	mdtable(header, rows) = Markdown.parse(join(
		["| " * join(r, " | ") * " |" for r in [header, fill("---", length(header)), rows...]], "\n"))
	function table(dir, name)
		data, header = readdlm(joinpath(dir, name), ','; header=true)
		keys = Tuple(Symbol.(vec(header)))
		cell(x) = x isa Number || x isa Bool ? x : something(tryparse(Float64, string(x)), string(x))
		[NamedTuple{keys}(Tuple(cell.(data[i, :]))) for i in axes(data, 1)]
	end
	istrue(x) = x === true || x == "true"
	ok(x) = istrue(x) ? "ok" : "**outside**"
	const results7 = joinpath(phase7, "results")
	pilot = table(results7, "pilot.csv")
	pilot_states = table(results7, "pilot_states.csv")
	suite7 = table(results7, "suite.csv")
	suite7_edges = table(results7, "suite_edges.csv")
	suite7_states = table(results7, "suite_states.csv")
	fits7 = table(results7, "fits.csv")
	worm_data = table(joinpath(phase7, "..", "phase5", "results"), "worm_data.csv")
end
#---
md"""
## 1. What the reading gave

- **Chossat & Krupa (2016)** write the Hopfield firing-rate equation on the cube ``[-1, 1]^n``, ``\dot x_j = (1 - x_j^2)\big(\lambda(Jx)_j + \lambda b_j - f(x_j)\big)``. Every face is invariant, so the ``2^n`` vertices are equilibria, and a cube edge whose ends have the right signs carries a heteroclinic connection. They build simple cycles of consecutive vertices with a learning rule. There is no branching and no control of how long each vertex holds.
- **The pseudoinverse papers** that follow it also store only simple cycles.
- **The latching-dynamics paper** needs fine tuning and gives no algorithm for a prescribed graph. The paper on consecutive and non-consecutive cycles was blocked (HAL) and isn't used.

What carries over is the cube and its invariant faces. The graph, the branching and the dwells come from the Phase 6 machinery.

## 2. The model

With ``x_j = \cos\psi_j`` and ``f(x) = x`` (their first-order regularisation), each neuron obeys

```math
\dot\psi_j = \sin\psi_j\,\big(g_j + k_j\cos\psi_j\big) + \eta\,\xi_j, \qquad g_j = -\Big(b_j + \sum_{m \ne j} J_{jm}\cos\psi_m\Big),
```

which is the Phase 6 equation with **pairwise** coupling only. At vertex ``v`` (every ``x_m = \pm1``) the rate along neuron ``j`` is

```math
r_j(v) = k_j - x_j\Big(b_j + \sum_{m\ne j} J_{jm}\,x_m\Big),
```

linear in neuron ``j``'s row ``(k_j, b_j, J_{j\cdot})``. So each neuron is designed on its own. Two things differ from Phase 6:

- **Departure and arrival are tied.** The two ends of a cube edge along ``j`` differ only in ``x_j``, so ``r_j(v) + r_j(\text{flip}_j v) = 2k_j``. Leaving at rate ``r`` means arriving at rate ``2k_j - r``. Setting ``k_j`` from the slowest departure along ``j`` (``k_j = (\text{slowest} - c)/2``) makes every arrival contract at least at ``c``, but faster departures along the same neuron arrive harder than that. The fit's race uses each edge's own arrival rate.
- **Not every pattern of signs is possible.** In Phase 6 every vertex rate was free. Here a neuron's rates across vertices come from one row, so the vertices where it flips must be separable by a threshold on the other neurons (a perceptron). A layout whose flips aren't of that kind has no design at all.

**Design.** For each neuron, one small quadratic program (JuMP + OSQP): every departure rate exactly, every other used vertex contracting along ``j`` by at least ``c/4`` and aimed at ``-c``, unused neighbours of the network pointed back toward it (softly), plus a small ridge.

**Layout.** First Phase 6's random search for the fewest neurons (up to 7), keeping only layouts whose every neuron's program is met. If none is met, a structured layout: one neuron per state (state ``i`` is the vertex with only neuron ``i`` on), edge ``i \to j`` switching ``j`` on then ``i`` off, and one relay neuron per two-way pair, so ``j \to i`` goes ``e_j \to e_j + e_r \to e_r \to e_r + e_i \to e_i``.

**Fit.** The Phase 6 race seed (`seed1d.jl`) with each exit's own arrival rate and noise ``\eta/\sqrt2``, then Phase 6's per-state correction passes, unchanged.
"""
#---
md"""
## 3. Precheck: which graphs have a design

Before any fit, every graph was laid out and designed at its rule-of-thumb rates and at its race seed. Every vertex's Jacobian was checked against the designed rates.
"""
#---
let
	lines = filter(l -> occursin(r"ε=|^3-ring", l), readlines(joinpath(results7, "precheck.log")))
	Markdown.parse("```\n" * join(lines, "\n") * "\n```")
end
#---
md"""
Every fixture graph has a design with 2 to 5 neurons, met to about ``10^{-13}``. The real worm graph had none in the search, up to 9 neurons. In a separate diagnosis 0 of 400 random layouts were met at each of 7, 9, 11 and 13 neurons, and even the bare sign pattern failed in every layout tried. Larger random cubes get less feasible, not more. The worm graph therefore uses the one-hot layout with relay neurons: 4 states and 4 two-way pairs make 8 neurons.
"""
#---
md"""
## 4. The pilot: a 3-state ring with dwells 1, 3 and 10
"""
#---
let
	rows = map(pilot) do r
		st = sort(filter(x -> x.noise == r.noise, pilot_states); by=x -> x.state)
		[r.noise, Int(r.neurons), join((round(x.set1; digits=3) for x in st), ", "), pct(r.worst_err), pct(r.rms, 2), pct(r.off, 2),
		 "$(istrue(r.set1_pass) ? "pass" : "fail") / $(istrue(r.set2_pass) ? "pass" : "fail") / $(istrue(r.half_pass) ? "pass" : "fail")",
		 istrue(r.third_pass) ? "pass" : "fail", join((round(x.cv; digits=2) for x in st), ", "), r.class]
	end
	mdtable(["Noise", "Neurons", "Dwells, set 1 (targets 1, 3, 10)", "Worst error, set 1", "RMS", "Off-ring", "Set 1 / set 2 / half step", "Third set", "Bout CV", "Class"], rows)
end
#---
md"""
**The pilot is a go.** Both noise levels pass all three checks and the information-only third set. Every dwell is within 0.6% of its target and nothing leaves the ring. The classifier calls both heteroclinic: all five starts end on saddles. With two neurons the design is met to ``10^{-14}`` and every vertex's Jacobian matches the designed rates to ``10^{-9}``. The race seed is close at noise 0.1. At ``10^{-3}`` it starts at about a third of the final rates (5.2 against 15.0 out of the 1 s state), and the correction passes close that within three passes.

The first ``10^{-3}`` run failed every check by 11–23% even though its own last pass hit the targets. It was a bug in resuming a fit across batches. The fit's asinh scale was recomputed from different starting rates, the Broyden update saw a spurious shift, and the final step went to its clamp. The scale is now saved with the checkpoint, and the row above is the refit. Phase 6 was not affected, because its suite recomputed the same seed in every batch.

## 5. The fixture graphs and the real worm
"""
#---
let
	why(r) = istrue(r.pass) ? "pass" : "fail: " * join(filter(!isempty, [istrue(r.dwell_ok) ? "" : "dwell", r.off >= 0.01 ? "off-graph" : "",
		istrue(r.branches_ok) ? "" : "branch", istrue(r.set2_pass) ? "" : "set 2", istrue(r.half_pass) ? "" : "half step"]), ", ")
	miss(r) = (e = filter(x -> x.case == r.case && x.noise == r.noise, suite7_edges); isempty(e) ? "–" :
		string(round(maximum(x -> max(abs(x.set1 - x.designed), abs(x.set2 - x.designed), abs(x.half_step - x.designed)), e); digits=3)))
	rows = [[r.case, r.noise, Int(r.neurons), r.layout, Int(r.transits), pct(r.worst_err), pct(r.rms, 2), pct(r.off, 2), miss(r), why(r), r.class, round(Int, r.seconds)] for r in suite7]
	mdtable(["Graph", "Noise", "Neurons", "Layout", "Transit vertices", "Worst dwell error, set 1", "RMS", "Off-graph", "Largest branch miss, any set", "Result", "Class", "Fit seconds"], rows)
end
#---
let
	n = length(suite7)
	passed = count(r -> istrue(r.pass), suite7)
	Markdown.parse("**$(passed) of $(n) cases pass all three checks.** Every branch on every check:")
end
#---
mdtable(["Graph", "Noise", "Edge", "Target", "Set 1", "Set 2", "Half step"],
	[[r.case, r.noise, "$(Int(r.from))→$(Int(r.to))", round(r.designed; digits=3), "$(round(r.set1; digits=3)) $(ok(r.set1_ok))",
	  "$(round(r.set2; digits=3)) $(ok(r.set2_ok))", "$(round(r.half_step; digits=3)) $(ok(r.half_step_ok))"] for r in suite7_edges])
#---
md"""
**What the failures are.** Every case whose fit ran to the end has every dwell within 1.8% of its target on every check, and most within 0.6%. Off-graph is at most 0.73%. Each of their failures is a branch-share bias of up to about one point on one state, and where it shows on more than one check it points the same way:
- foraging at ``10^{-3}``: 2→3 is 0.707 against 0.70 on set 2 only;
- shared-conflicting at ``10^{-3}``: 4→2 is 0.409 and 0.412 against 0.40 on set 1 and the half step;
- the real worm graph, below.

That is the same kind of miss Phase 6 left: 9 of 13 pass here, against Phase 6's 8 of 13 under its old seed.

**Shared-solvable at noise 0.1 reaches the limit of the pairwise design.** State 6 (0.7 s) leaves through two transit vertices, and at this noise those alone take nearly its whole dwell. The fit raised its one exit rate without bound (past 400) until no design existed. Capping out-rates at twice the contraction (agreed 2026-11-13) stopped that. The cap is above every other case's final rates: the highest is 1.14 times the contraction. But as the rates out of state 4 then rose, neuron 2's program had no solution at pass 5, so pairwise coupling can't hold those rates on this layout. The row checks pass 4, the last pass whose network could be built: state 6 is 22% long, and states 1 and 4 miss their branch shares by 1–2 points. Its fit time counts only the final batch. Phase 6 also failed this graph at 0.1 but passed it at ``10^{-3}``, as here.

### The real worm

The measured graph from Phase 5 (122 worms, every transition seen in the tracks), against the fit at noise 0.1.
"""
#---
let
	names = WORM_STATES
	d(i) = only(filter(x -> x.kind == "dwell" && Int(x.from) == i, worm_data)).value
	st = sort(filter(x -> x.case == "worm data", suite7_states); by=x -> x.state)
	isempty(st) && return md"The worm-data fit hasn't finished."
	e = filter(x -> x.case == "worm data", suite7_edges)
	A = mdtable(["State", "Tracks (s)", "Set 1", "Set 2", "Half step"],
		[[names[i], round(d(i); digits=3), "$(round(st[i].set1; digits=3)) ($(pct(st[i].set1 / d(i) - 1)))",
		  "$(round(st[i].set2; digits=3)) ($(pct(st[i].set2 / d(i) - 1)))", "$(round(st[i].half_step; digits=3)) ($(pct(st[i].half_step / d(i) - 1)))"] for i in 1:4])
	B = mdtable(["Edge", "Tracks", "Set 1", "Set 2", "Half step"],
		[["$(names[Int(r.from)]) → $(names[Int(r.to)])", round(r.designed; digits=4), "$(round(r.set1; digits=3)) $(ok(r.set1_ok))",
		  "$(round(r.set2; digits=3)) $(ok(r.set2_ok))", "$(round(r.half_step; digits=3)) $(ok(r.half_step_ok))"] for r in e])
	md"""
	$A

	$B
	"""
end
#---
md"""
**The real worm graph nearly passes, as it did in Phase 6.** The search found no layout up to 9 neurons, so it uses one-hot with relays: 8 neurons and 17 transit vertices. All four mean bouts are within 0.9% on every check with nothing off-graph, and set 1 passes on every branch. Set 2 misses Pause 1's split (0.476 against 0.469 to Forward) and Reverse → Forward (0.0076 against 0.0056). The half step misses Forward → Reverse (0.0073 against 0.0098). Four of the five classifier starts settle on a vertex, so the label is `unknown`. Forward's and Reverse's exits fitted to negative rates, which puts those vertices on the excitable side, as in Phase 6's race-seed fit. The fit took 53 minutes.

## 6. Watch one

The fitted network, simulated live: ``x_j = \cos\psi_j`` for each neuron (``+1`` and ``-1`` are its two vertex values), then the decoded state.
"""
#---
@bind hop_choice Select([i => "$(suite7[i].case), noise $(suite7[i].noise)" for i in eachindex(suite7)]; default=1)
#---
@bind hop_seed Select([1 => "seed 1", 2 => "seed 2", 3 => "seed 3"]; default=1)
#---
let
	r = suite7[hop_choice]
	cases = [[(c[1], c[2], c[3]) for c in NAMED]; [worm_data_case()]]
	_, P, τ = only(c for c in cases if c[1] == r.case)
	spec = (; P=float(P), τ, form=:u, ε₀=r.noise, sample=startswith(r.case, "worm") ? WORM_SPEC.sample : minimum(τ) / 10)
	L = hop_layout(spec)                              # the search is seeded, so this is the suite's layout
	rates = sort(filter(x -> x.case == r.case && x.noise == r.noise, fits7); by=x -> x.edge)
	m, _ = hop_model(spec, L, [Float64(x.out) for x in rates]; fast=rates[1].fast, contract=rates[1].contract)
	T = min(10sum(τ), 60.0)
	every, dt = frame_steps(m, spec.sample, 0.02)
	sol = solve(SDEProblem(drift_hop, noise_flat, hop_start(m), (0.0, T), hop_params(m); seed=UInt64(hop_seed)), EM();
		dt, adaptive=false, saveat=spec.sample, maxiters=typemax(Int))
	names = startswith(r.case, "worm") ? WORM_STATES : ["state $i" for i in 1:size(P, 1)]
	top = plot(; ylabel="x = cos ψ", title="$(r.case), noise $(r.noise): $(m.n) neurons", titlefontsize=11, legend=:outerright, xticks=false, ylims=(-1.1, 1.1))
	for j in 1:m.n
		plot!(top, sol.t, [cos(ψ[j]) for ψ in sol.u]; lw=1.0, color=state_colors[mod1(j, 6)], linestyle=j > 6 ? :dash : :solid, label="neuron $j")
	end
	decoded = Int[]
	for ψ in sol.u
		v = 1 + sum((cos(ψ[j]) < 0) << (j - 1) for j in 1:m.n)
		k = m.node_of[v]
		push!(decoded, k == 0 ? (isempty(decoded) ? 1 : decoded[end]) : k)
	end
	bottom = plot(sol.t, decoded; seriestype=:steppost, color=muted, lw=1.2, label="", xlabel="time (s)",
		yticks=(1:size(P, 1), names), ylims=(0.5, size(P, 1) + 0.5))
	plot(top, bottom; layout=Plots.grid(2, 1; heights=[0.6, 0.4]), size=(820, 450))
end
#---
md"""
## 7. What Phase 7 hasn't measured

- **Dwell variability.** Only means are fitted.
- **Higher-order regularisations.** Only ``f(x) = x`` was used. Chossat & Krupa's higher Taylor terms of ``\operatorname{arctanh}`` change the flow inside the cube, not the vertex rates.
- **Noise on ``x``.** Noise acts on the angle; additive noise on ``x`` itself would leave the cube.
- **Larger graphs.** One-hot with relays grows as states plus two-way pairs; the search was only tried up to 9 neurons.
- **Layouts that fit the rates.** A layout is accepted on its rule-of-thumb rates; the fitted rates can leave its design (shared-solvable at 0.1). Re-searching with the fitted rates, or weighting transits by the dwell of the state they belong to, wasn't tried.
- The package still has no `src/`.
"""
