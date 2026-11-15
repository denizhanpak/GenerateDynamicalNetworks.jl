# The reproduction subset: each backend refitted with the prototype's settings on one branching graph (Δ-clique at
# noise 0.1), and synchrony and Hopfield on the pilot (3-ring, dwells 1, 3, 10, both noise levels), checked as the
# prototype checked them and compared with its stored tables (research/benchmarks/phase*/results).
#
# Where the random-number path is unchanged the numbers must be equal (`exact`). Elsewhere (`band`) the refit is a
# different model, so its dwells cannot match the stored ones to sampling error; instead every state's mean dwell
# must fall on the same side of the acceptance band (within 5% of the target) as the stored one, on each dwell
# check the stored table has (set 1, set 2, half step). This covers Hopfield, whose OSQP step-size interval is now
# fixed, and the synchrony pilot, whose special 3-ring layout was not ported.
#
#   julia --project=benchmarks -t 7 benchmarks/reproduce.jl [--only=backend] [--expect=exact|band]
#
# Writes results/reproduction.csv: backend, case, noise, expect, agree, stored_pass, new_pass, fit_seconds,
# check_seconds, date, details. A case run again replaces its earlier row.
include(joinpath(@__DIR__, "common.jl"))

const PROTO = joinpath(@__DIR__, "..", "research", "benchmarks")
proto(phase, file) = joinpath(PROTO, phase, "results", file)
const REPRO_CSV = joinpath(RESULTS, "reproduction.csv")

"""The rows of a stored table as Dicts by column name, keeping those whose named columns have the given values."""
function stored_rows(file; match...)
    header = split(readline(file), ",")
    rows = [Dict(zip(header, r)) for r in read_rows(file)]
    return filter(r -> all(k -> (v = r[String(k)]; v == string(match[k]) || tryparse(Float64, v) == match[k]), keys(match)), rows)
end

# protocol :repeat — validate on set 1, and the branch check again on set 2 (phases 3–5);
# protocol :accept — the three checks (phases 6–7). `states` is the stored per-state table, if any.
const SUBSET = vcat(
    [(key="horchler", backend=Horchler(; seconds=Inf), name="Δ-clique", ε=0.1, protocol=:repeat, expect=:exact,
      row=() -> only(stored_rows(proto("phase3", "branching.csv"); case="Δ-clique", noise=0.1)), states=nothing),
     (key="simplex", backend=Simplex(), name="Δ-clique", ε=0.1, protocol=:repeat, expect=:exact,
      row=() -> only(stored_rows(proto("phase4", "suite.csv"); case="Δ-clique", noise=0.1, backend="simplex")), states=nothing),
     (key="cylinder", backend=Cylinder(), name="Δ-clique", ε=0.1, protocol=:repeat, expect=:exact,
      row=() -> only(stored_rows(proto("phase4", "suite.csv"); case="Δ-clique", noise=0.1, backend="cylinder")), states=nothing),
     (key="twopop", backend=TwoPopulation(; seconds=120), name="Δ-clique", ε=0.1, protocol=:repeat, expect=:exact,
      row=() -> only(stored_rows(proto("phase5", "suite.csv"); case="Δ-clique", noise=0.1, backend="two", regime="default")),
      states=nothing),
     (key="ctrnn", backend=CTRNNDesign(; by_state=false), name="Δ-clique", ε=0.1, protocol=:repeat, expect=:exact,
      row=() -> only(stored_rows(proto("phase5", "suite.csv"); case="Δ-clique", noise=0.1, backend="ctrnn", regime="design")),
      states=nothing),
     (key="synchrony", backend=Synchrony(), name="Δ-clique", ε=0.1, protocol=:accept, expect=:exact,
      row=() -> only(stored_rows(proto("phase6", "suite.csv"); case="Δ-clique", noise=0.1)), states=nothing),
     (key="hopfield", backend=Hopfield(), name="Δ-clique", ε=0.1, protocol=:accept, expect=:band,
      row=() -> only(stored_rows(proto("phase7", "suite.csv"); case="Δ-clique", noise=0.1)),
      states=() -> stored_rows(proto("phase7", "suite_states.csv"); case="Δ-clique", noise=0.1))],
    [(key="synchrony", backend=Synchrony(), name=PILOT, ε, protocol=:accept, expect=:band,
      row=() -> only(stored_rows(proto("phase6", "pilot.csv"); noise=ε)),
      states=() -> stored_rows(proto("phase6", "pilot_states.csv"); noise=ε)) for ε in (1e-3, 0.1)],
    [(key="hopfield", backend=Hopfield(), name=PILOT, ε, protocol=:accept, expect=:band,
      row=() -> only(stored_rows(proto("phase7", "pilot.csv"); noise=ε)),
      states=() -> stored_rows(proto("phase7", "pilot_states.csv"); noise=ε)) for ε in (1e-3, 0.1)])

"""The new model's numbers, checked as the prototype checked them, under the stored tables' column names, and its
mean dwells per check under the stored per-state tables' column names."""
function check_new(model, spec, protocol)
    if protocol === :repeat
        v1 = validate(model, spec)
        v2 = validate(model, spec; seeds=9_200_001:9_200_064)
        return Dict("set1" => v1.dwell, "set2" => v2.dwell),
               Dict{String,Any}("worst_err" => maximum(abs, v1.err), "rms" => v1.rms, "off" => v1.off,
                                "dwell_ok" => v1.dwell_ok, "branches_ok" => v1.branch_ok,
                                "repeat_branches_ok" => v2.branch_ok, "pass" => v1.pass)
    end
    a = accept(model, spec)
    s1 = a.set1
    return Dict("set1" => s1.dwell, "set2" => a.set2.dwell, "half_step" => a.half_step.dwell),
           Dict{String,Any}("worst_err" => maximum(abs, s1.err), "rms" => s1.rms, "off" => s1.off, "dwell_ok" => s1.dwell_ok,
                            "branches_ok" => s1.branch_ok, "set1_pass" => s1.pass, "set2_pass" => a.set2.pass,
                            "half_pass" => a.half_step.pass, "pass" => a.pass)
end

const TOL = 0.05   # the acceptance band: mean dwell within 5% of the target

"""Compare the stored and new mean dwells state by state: in or out of the acceptance band, on each dwell check the
stored per-state table has. Returns whether every verdict matches, and the relative errors (stored/new, %)."""
function band_agreement(st, dwells, τ)
    checks = [k for k in ("set1", "set2", "half_step") if haskey(first(st), k)]
    parts, mismatch = String[], String[]
    for k in checks
        old = [parse(Float64, r[k]) / τ[j] - 1 for (j, r) in enumerate(st)]
        new = dwells[k] ./ τ .- 1
        push!(parts, "$k errors % stored/new: " * join((@sprintf("%.2f/%.2f", 100a, 100b) for (a, b) in zip(old, new)), " "))
        append!(mismatch, ("$k state $j" for j in eachindex(old) if (abs(old[j]) <= TOL) != (abs(new[j]) <= TOL)))
    end
    return isempty(mismatch), join(parts, "; ") * "; band verdicts that differ: " * (isempty(mismatch) ? "none" : join(mismatch, ", "))
end

"""Append a row to the reproduction table, replacing any earlier row for the same case."""
function write_result(header, row)
    if isfile(REPRO_CSV)
        key = case_key(row[1], row[2], row[3])
        lines = readlines(REPRO_CSV)
        keep = [lines[1]; [l for (l, r) in zip(lines[2:end], read_rows(REPRO_CSV)) if case_key(r[1], r[2], parse(Float64, r[3])) != key]]
        length(keep) < length(lines) && write(REPRO_CSV, join(keep, "\n") * "\n")
    end
    append_row(REPRO_CSV, header, row)
end

same_value(stored::AbstractString, new::Bool) = stored == string(new)
same_value(stored::AbstractString, new::Real) = parse(Float64, stored) == new

function main(args)
    opt(name) = (i = findfirst(startswith("--$name="), args); i === nothing ? nothing : split(args[i], "="; limit=2)[2])
    only_key, expect = opt("only"), opt("expect")
    for c in SUBSET
        only_key === nothing || c.key == only_key || continue
        expect === nothing || String(c.expect) == expect || continue
        spec = case_spec(c.name, c.ε)
        t0 = time()
        model = fit(spec, c.backend)
        fit_seconds = time() - t0
        t1 = time()
        dwells, new = check_new(model, spec, c.protocol)
        check_seconds = time() - t1
        stored = c.row()
        compared = sort([k for k in keys(new) if haskey(stored, k)])
        differ = [k for k in compared if !same_value(stored[k], new[k])]
        stored_pass = haskey(stored, "pass") ? stored["pass"] == "true" :
                      all(stored[k] == "true" for k in ("set1_pass", "set2_pass", "half_pass"))
        if c.expect === :exact
            agree = isempty(differ)
            details = agree ? "equal: $(join(compared, ", "))" :
                      join(("$k stored $(stored[k]) new $(new[k])" for k in differ), "; ")
        else
            st = sort(c.states(); by=r -> parse(Int, r["state"]))
            length(st) == nstates(spec) && all(j -> parse(Float64, st[j]["target"]) == spec.τ[j], eachindex(st)) ||
                error("the stored per-state table does not match $(c.name) at ε=$(c.ε)")
            agree, band = band_agreement(st, dwells, spec.τ)
            details = band * "; numbers that differ: " * (isempty(differ) ? "none" : join(differ, ", "))
        end
        write_result(["backend", "case", "noise", "expect", "agree", "stored_pass", "new_pass", "fit_seconds",
                      "check_seconds", "date", "details"],
                     (c.key, c.name, c.ε, c.expect, agree, stored_pass, new["pass"], fit_seconds, check_seconds, now(), details))
        @printf("  %s %s ε=%g (%s): %s; pass stored %s, new %s; fit %.0f s, checks %.0f s\n    %s\n", c.key, c.name, c.ε,
                c.expect, agree ? "AGREES" : "DIFFERS", stored_pass, new["pass"], fit_seconds, check_seconds, details)
        flush(stdout)
    end
    return 0
end

exit(main(ARGS))
