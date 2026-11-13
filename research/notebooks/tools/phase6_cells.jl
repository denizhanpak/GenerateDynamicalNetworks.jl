begin
	import Pkg
	Pkg.activate(@__DIR__)   # notebooks/Project.toml
	Pkg.instantiate()
	using Plots, PlutoUI, DelimitedFiles, Printf
	# The packages the included files load. Pluto only re-imports what it sees in a cell, so they
	# are listed here too, or a cell rerun after a control moves can't find `solve` or `remake`.
	using SHCTools, StochasticDiffEq, StaticArrays, Graphs, Distributions, QuadGK, Roots
	using LinearAlgebra, Random, Statistics, Serialization
	import CMAEvolutionStrategy as CES
	const phase6 = joinpath(@__DIR__, "..", "benchmarks", "phase6")
	include(joinpath(phase6, "synchrony.jl"))      # the backend, with Phases 2 to 5 (pipeline, classifier) through it
	include(joinpath(phase6, "..", "phase5", "twoway_cases.jl"))   # the worm graph measured from the tracks
end
#---
md"""
# GenerateDynamicalNetworks: Phase 6, phase oscillators in synchrony space

The go/no-go phase. The plan: identical phase oscillators whose synchrony subspaces stay exactly invariant, each graph state a cluster pattern read from phase differences, a 3-state ring with unequal dwell as the pilot, and, if the pilot can't reach tolerance, a uniform-dwell backend marked experimental.

Choices you made for this phase (2026-11-11):

| Choice | Setting |
|:--|:--|
| Oscillators | pairs of oscillators with one common frequency; coupling coefficients set per pair, as in Bick & Lohse (2019) |
| Coupling | terms built from products of the other pairs' ``\cos\psi``, so every corner's rate in every direction is set independently |
| States | each graph state a corner of the S/D cube; each edge a path of single-pair flips through hidden transit corners, whose time counts toward the state being left |
| Fit, noise, scope | passage-law seed, per-state correction, one shared noise at ``10^{-3}`` and ``0.1``; pilot first, then the six fixture graphs and the real worm graph |

The code is in `benchmarks/phase6/`: `synchrony.jl` (model, layouts, fits), `pilot.jl` and `suite.jl`.
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
	const results6 = joinpath(phase6, "results")
	pilot = table(results6, "pilot.csv")
	pilot_states = table(results6, "pilot_states.csv")
	suite6 = table(results6, "suite.csv")
	suite6_states = table(results6, "suite_states.csv")
	suite6_edges = table(results6, "suite_edges.csv")
	fits6 = table(results6, "fits.csv")
	worm_data = table(joinpath(phase6, "..", "phase5", "results"), "worm_data.csv")
end
#---
md"""
## 1. The model

``2M`` oscillators with one natural frequency, in ``M`` pairs. Pair ``\sigma`` has phase difference ``\psi_\sigma``: in phase (S, ``\psi_\sigma = 0``) or anti-phase (D, ``\psi_\sigma = \pi``). Each oscillator of pair ``\sigma`` obeys

```math
\dot\theta_{\sigma,k} = \omega + a_\sigma \sin\Delta_k + b_\sigma \sin 2\Delta_k + \eta\,\xi, \qquad \Delta_k = \theta_{\sigma,\text{other}} - \theta_{\sigma,k},
```

with ``a_\sigma, b_\sigma`` multilinear in ``\cos\psi_\tau`` of the other pairs, so the coupling involves several oscillators at once (Bick 2018, 2019). The phase differences then obey

```math
\dot\psi_\sigma = \sin\psi_\sigma\,\big(g_\sigma + k_\sigma\cos\psi_\sigma\big), \qquad g = -2a, \; k = -4b .
```

Every pair's S and D sets are exactly invariant (the swap within the pair), and the ``2^M`` S/D patterns, the corners of a cube, are equilibria. At corner ``v`` the rate in direction ``\sigma`` is ``\lambda_\sigma(v) = g_\sigma + k_\sigma x_\sigma`` with ``x = \pm 1``. Setting ``g_\sigma`` and ``k_\sigma`` at each pattern of the other pairs sets every ``\lambda_\sigma(v)`` independently, and a cube edge whose ends have opposite signs carries a one-dimensional connection (Bick's Remark 3.3, extended). The pilot checked that each corner's Jacobian has exactly the designed rates, to ``10^{-9}``.

**Layout.** Each graph state is a corner and each edge a path of single-pair flips through hidden transit corners. Edges leaving one state start in different directions, no corner serves two paths, and a two-way pair needs two separate paths. `sync_layout` searches random placements for the fewest pairs, then the fewest transits. Unused corners point toward the network. The decoder reads the corner from the signs of ``\cos\psi``; time at a transit counts toward the state being left. Runs use the frame rotating with the common frequency (``\omega = 0``), which changes no phase difference.

**Fit.** The passage law ``T \approx \ln(1/\eta)/\lambda`` seeds each out-rate. Correction passes then take one Broyden step per state on ``\operatorname{asinh}(\lambda_e/c)`` over its out-edges against its log dwell and log branch ratios. A rate may turn negative: the corner is then weakly stable and noise has to cross a small barrier inside the edge (the excitable side).

## 2. What the reading showed

- **Fully identical coupling can't give unequal dwells.** One all-to-all population (Ashwin, Orosz, Wordsworth & Townley 2007) has 30 interchangeable saddles for 5 oscillators, all with the same rates, and noise wanders over all of them.
- **Synchrony from network structure** (Field 2015; Aguiar, Ashwin, Dias & Field 2011) does allow rates per node, but those constructions are general identical cells with bump functions, not phase oscillators. Their full texts were blocked; this rests on Garrido da Silva & Soares (2026), which reproduces Field's construction, and on Agarwal's 2011 thesis.
- **Bick's groups of two** keep each group's S and D states invariant through the swap within the group, and Bick & Lohse (2019) break the symmetry between groups on purpose. That is the route taken here.
- No paper found builds a prescribed graph with prescribed unequal dwells from phase oscillators.

## 3. The pilot: a 3-state ring with dwells 1, 3 and 10

Two pairs, four oscillators: the states sit on SS, DS and DD, and the transit SD follows the longest dwell, since a cube cycle has even length.
"""
#---
let
	rows = map(pilot) do r
		st = sort(filter(x -> x.noise == r.noise, pilot_states); by=x -> x.state)
		[r.noise, join((round(x.set1; digits=3) for x in st), ", "), pct(r.worst_err), pct(r.rms, 2), pct(r.off, 2),
		 "$(istrue(r.set1_pass) ? "pass" : "fail") / $(istrue(r.set2_pass) ? "pass" : "fail") / $(istrue(r.half_pass) ? "pass" : "fail")",
		 istrue(r.third_pass) ? "pass" : "fail", join((round(x.cv; digits=2) for x in st), ", "), r.class, round(Int, r.seconds)]
	end
	mdtable(["Noise", "Dwells, set 1 (targets 1, 3, 10)", "Worst error, set 1", "RMS", "Off-ring", "Set 1 / set 2 / half step", "Third set", "Bout CV", "Class", "Fit seconds"], rows)
end
#---
md"""
**The pilot is a go.** At noise 0.1 the first version failed: the 10 s state topped out near 6 s with its rate pushed to almost zero, because even at ``\lambda = 0`` the flow away from a corner grows like the cube of the distance, and at that noise the cubic push alone ends a visit in about 6 s. Letting rates turn negative fixed it. That corner is then excitable while the other two stay heteroclinic, which is why the classifier's starts disagree and the label is `unknown`. Its dwell is exponentially sensitive to the rate, so the general fit takes smaller steps in its last passes.

## 4. The fixture graphs and the real worm
"""
#---
let
	why(r) = istrue(r.pass) ? "pass" : "fail: " * join(filter(!isempty, [istrue(r.dwell_ok) ? "" : "dwell", r.off >= 0.01 ? "off-graph" : "",
		istrue(r.branches_ok) ? "" : "branch", istrue(r.set2_pass) ? "" : "set 2", istrue(r.half_pass) ? "" : "half step"]), ", ")
	miss(r) = (e = filter(x -> x.case == r.case && x.noise == r.noise, suite6_edges); isempty(e) ? "–" :
		string(round(maximum(x -> max(abs(x.set1 - x.designed), abs(x.set2 - x.designed), abs(x.half_step - x.designed)), e); digits=3)))
	rows = [[r.case, r.noise, "$(Int(r.oscillators)) ($(Int(r.pairs)) pairs)", Int(r.transits), pct(r.worst_err), pct(r.rms, 2), pct(r.off, 2), miss(r), why(r), r.class, round(Int, r.seconds)] for r in suite6]
	mdtable(["Graph", "Noise", "Oscillators", "Transit corners", "Worst dwell error, set 1", "RMS", "Off-graph", "Largest branch miss, any set", "Result", "Class", "Fit seconds"], rows)
end
#---
let
	n = length(suite6)
	passed = count(r -> istrue(r.pass), suite6)
	Markdown.parse("**$(passed) of $(n) cases pass all three checks.** Every branch on every check:")
end
#---
mdtable(["Graph", "Noise", "Edge", "Target", "Set 1", "Set 2", "Half step"],
	[[r.case, r.noise, "$(Int(r.from))→$(Int(r.to))", round(r.designed; digits=3), "$(round(r.set1; digits=3)) $(ok(r.set1_ok))",
	  "$(round(r.set2; digits=3)) $(ok(r.set2_ok))", "$(round(r.half_step; digits=3)) $(ok(r.half_step_ok))"] for r in suite6_edges])
#---
md"""
**What the failures on the fixture graphs are.** Every dwell on every fixture graph is within 1.6% of its target on all three checks. Each failing case misses on one state's branch shares, by 0.4 to 1.2 percentage points. Where a miss shows on more than one check it points the same way: figure-eight 1→2 measures 0.756, 0.753 and 0.757 against 0.75. So it's a small bias left by the fit, not chance in the check. At noise ``10^{-3}`` a 64-run check sees so many exits that its 95% interval is narrower than that. A larger last fit pass, or averaging the last passes, might close the gap; neither was tried. All 13 cases were fitted under one rule: one contraction strength everywhere, and the same seed and correction passes.

### The real worm

The measured graph from Phase 5 (122 worms, every transition seen in the tracks), against the fit at noise 0.1.
"""
#---
let
	names = WORM_STATES
	d(i) = only(filter(x -> x.kind == "dwell" && Int(x.from) == i, worm_data))
	st = sort(filter(x -> x.case == "worm data", suite6_states); by=x -> x.state)
	isempty(st) ? md"The worm-data fit hasn't finished." :
	mdtable(["State", "Mean bout, tracks (s)", "Model set 1", "Set 2", "Half step"],
		[[names[i], round(d(i).value; digits=3), round(st[i].set1; digits=3), round(st[i].set2; digits=3), round(st[i].half_step; digits=3)] for i in 1:4])
end
#---
let
	names = WORM_STATES
	e = filter(x -> x.case == "worm data", suite6_edges)
	isempty(e) ? md"" :
	mdtable(["Edge", "Tracks", "Model set 1", "Set 2", "Half step"],
		[["$(names[Int(r.from)]) → $(names[Int(r.to)])", round(r.designed; digits=3), round(r.set1; digits=3),
		  round(r.set2; digits=3), round(r.half_step; digits=3)] for r in e])
end
#---
md"""
**The real worm graph fails.** Pause 1 and Pause 2 are right, but Forward reaches only about half its 5.7 s and Reverse about two thirds of its 2.1 s. Forward also leaves to Reverse 29% of the time instead of 1%. The fit had pushed the Forward → Reverse rate below zero (−0.71), but with this contraction strength and noise, a negative rate puts up almost no barrier. The contraction is set by the fastest state, Pause 2 at 0.22 s, and that same contraction sets how sharply the flow leaves every corner. Rare exits from slow states, the 1% and 0.6% branches here, are where this design struggles.

Scaling each arrival to its own departure rate was tried and dropped: the dwells swung between 0.4 and 10 s from pass to pass.

### Option B: start the fit from a 1-D race (time-boxed, real worm graph only)

Each exit from a state corner is close to a 1-D diffusion of one phase difference, ``d\psi = \sin\psi\,(g + k\cos\psi)\,dt + \sqrt2\,\eta\,dW`` with ``g + k = \lambda`` and ``k - g = -\text{contract}``. The state ends when ``|\psi|`` reaches ``\pi/2``. `benchmarks/phase6/seed1d.jl` computes each exit's survival on a grid and races a state's exits against each other independently. That gives the dwell and shares any set of rates should produce. It then solves for the rates that give the target dwell and shares, and starts the correction passes there, with their starting matrix taken from the same race.

Using only each exit's mean time, treated as exponential, fixed Forward and Reverse (5.61 s and 2.08 s, Forward → Reverse 0.009). It failed on Pause 2: at 0.22 s its main exit fires almost on schedule, so the 5% exit never won (0 of thousands) and its dwell was 24% too long. Using the full waiting-time distribution fixed that. On the earlier fit's rates the race reproduces what was measured, including Pause 2 never reaching Reverse; for Pause 1 it predicts 0.80 / 0.20 against 0.48 / 0.52 measured. Below are the results of the full-race fit, with the same one contraction everywhere, the same correction passes and the same checks as the suite.
"""
#---
let
	dir = joinpath(results6, "seed1d")
	isfile(joinpath(dir, "suite_states.csv")) || return md"The option B fit hasn't finished."
	names = WORM_STATES
	st = sort(filter(x -> x.case == "worm data", table(dir, "suite_states.csv")); by=x -> x.state)
	e = filter(x -> x.case == "worm data", table(dir, "suite_edges.csv"))
	d(i) = only(filter(x -> x.kind == "dwell" && Int(x.from) == i, worm_data)).value
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
**With the race seed the real worm graph nearly passes.** All four mean bouts are within 0.9% on every check, with 0 off-graph, and set 1 and the half step pass on every branch. Set 2 misses one branch: Forward → Reverse at 0.012 against 0.0098. That's the same kind of narrow single-branch miss the CTRNN design had on this graph (Pause 2 → Forward there). The classifier calls this fit excitable: every start settles. 12 oscillators, 11 transit corners, fit 21 minutes.

### All 13 cases under the race seed (2026-11-13)

The same suite rerun with every fit started from the race seed, using the same layouts, correction passes, contraction rule and checks. The race solve now starts from the rule-of-thumb rates. The mean-only start stuck at the rate clamp on the worm ring at ``10^{-3}``, and in every other case it gives the same seed to 4 significant figures, so the real worm result above is kept.
"""
#---
let
	dir = joinpath(results6, "seed1d")
	isfile(joinpath(dir, "suite.csv")) || return md"The race-seed rerun hasn't finished."
	new = table(dir, "suite.csv")
	edges = table(dir, "suite_edges.csv")
	why(r) = istrue(r.pass) ? "pass" : "fail: " * join(filter(!isempty, [istrue(r.dwell_ok) ? "" : "dwell", r.off >= 0.01 ? "off-graph" : "",
		istrue(r.branches_ok) ? "" : "branch", istrue(r.set2_pass) ? "" : "set 2", istrue(r.half_pass) ? "" : "half step"]), ", ")
	old(r) = (o = filter(x -> x.case == r.case && x.noise == r.noise, suite6); isempty(o) ? "–" : (istrue(only(o).pass) ? "pass" : "fail"))
	miss(r) = (e = filter(x -> x.case == r.case && x.noise == r.noise, edges); isempty(e) ? "–" :
		string(round(maximum(x -> max(abs(x.set1 - x.designed), abs(x.set2 - x.designed), abs(x.half_step - x.designed)), e); digits=3)))
	A = mdtable(["Graph", "Noise", "Old seed", "Race seed", "Worst dwell error, set 1", "Off-graph", "Largest branch miss, any set", "Class", "Fit seconds"],
		[[r.case, r.noise, old(r), why(r), pct(r.worst_err), pct(r.off, 2), miss(r), r.class, round(Int, r.seconds)] for r in new])
	B = mdtable(["Graph", "Noise", "Edge", "Target", "Set 1", "Set 2", "Half step"],
		[[r.case, r.noise, "$(Int(r.from))→$(Int(r.to))", round(r.designed; digits=4), "$(round(r.set1; digits=4)) $(ok(r.set1_ok))",
		  "$(round(r.set2; digits=4)) $(ok(r.set2_ok))", "$(round(r.half_step; digits=4)) $(ok(r.half_step_ok))"]
		 for r in edges if !(istrue(r.set1_ok) && istrue(r.set2_ok) && istrue(r.half_step_ok))])
	passed = count(r -> istrue(r.pass), new)
	md"""
	**$(passed) of $(length(new)) pass under the race seed.**

	$A

	The branches that miss on at least one check:

	$B
	"""
end
#---
md"""
**The race seed doesn't raise the pass count: 6 of 13, against 8 under the old seed.** It does tighten the dwells: every dwell is within 1.8% on every check, and nothing is above 0.2% off-graph. Every failure is a single state's branch shares on one check, each within about one percentage point. Unlike the old seed's misses, these mostly change sign from one check to the next. Foraging at ``10^{-3}`` measures 2→3 at 0.707 on set 1 and 0.694 on set 2, against 0.70. Δ-clique at 0.1 misses only on set 1 (0.390 against 0.40) and is exact at the half step. That looks like chance, not bias.

That is expected from how strict the check is. Each branch must sit inside its own 95% interval on three checks. So a model whose shares are exactly right would still fail now and then: the interval is missed 5% of the time by chance, and a case has several branches. Which cases pass at this level is partly luck, and the 8 against 6 is within that. Wider intervals or a corrected threshold would change the counts, but the acceptance rule was left as agreed.
"""
#---
md"""
## 5. Watch one

The fitted model, simulated live: ``\cos\psi_\sigma`` for each pair (``+1`` in phase, ``-1`` anti-phase), then the decoded state.
"""
#---
@bind sync_choice Select([i => "$(suite6[i].case), noise $(suite6[i].noise)" for i in eachindex(suite6)]; default=1)
#---
@bind sync_seed Select([1 => "seed 1", 2 => "seed 2", 3 => "seed 3"]; default=1)
#---
let
	r = suite6[sync_choice]
	cases = [[(c[1], c[2], c[3]) for c in NAMED]; [worm_data_case()]]
	_, P, τ = only(c for c in cases if c[1] == r.case)
	spec = (; P=float(P), τ, form=:u, ε₀=r.noise, sample=startswith(r.case, "worm") ? WORM_SPEC.sample : minimum(τ) / 10)
	L = sync_layout(spec.P)                           # the search is seeded, so this is the suite's layout
	rates = sort(filter(x -> x.case == r.case && x.noise == r.noise, fits6); by=x -> x.edge)
	m = sync_model(spec, L, [Float64(x.out) for x in rates]; fast=rates[1].fast, contract=rates[1].contract,
		arrive=rates[1].arrive, floor=rates[1].floor)
	T = min(10sum(τ), 60.0)
	every, dt = frame_steps(m, spec.sample, 0.02)
	sol = solve(SDEProblem(drift_sync, noise_flat, sync_start(m), (0.0, T), sync_params(m); seed=UInt64(sync_seed)), EM();
		dt, adaptive=false, saveat=spec.sample, maxiters=typemax(Int))
	names = startswith(r.case, "worm") ? WORM_STATES : ["state $i" for i in 1:size(P, 1)]
	top = plot(; ylabel="cos ψ", title="$(r.case), noise $(r.noise): $(2L.M) oscillators", titlefontsize=11, legend=:outerright, xticks=false, ylims=(-1.1, 1.1))
	for σ in 1:L.M
		plot!(top, sol.t, [cos(θ[2σ] - θ[2σ-1]) for θ in sol.u]; lw=1.0, color=state_colors[mod1(σ, 6)], label="pair $σ")
	end
	decoded = Int[]
	for θ in sol.u
		v = 1 + sum((cos(θ[2σ] - θ[2σ-1]) < 0) << (σ - 1) for σ in 1:L.M)
		k = m.node_of[v]
		push!(decoded, k == 0 ? (isempty(decoded) ? 1 : decoded[end]) : k)
	end
	bottom = plot(sol.t, decoded; seriestype=:steppost, color=muted, lw=1.2, label="", xlabel="time (s)",
		yticks=(1:size(P, 1), names), ylims=(0.5, size(P, 1) + 0.5))
	plot(top, bottom; layout=Plots.grid(2, 1; heights=[0.6, 0.4]), size=(820, 450))
end
#---
md"""
## 6. What Phase 6 hasn't measured

- **Dwell variability.** Only means are fitted, and heteroclinic corners give nearly deterministic dwells.
- **Larger graphs.** The layout search is random and the cube grows as ``2^M``; the real worm graph already needs 12 oscillators.
- The package still has no `src/`. Phase 7 (Hopfield) is in `phase7_hopfield.jl`.
"""
