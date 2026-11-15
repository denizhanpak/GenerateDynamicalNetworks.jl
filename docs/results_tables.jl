# The tables on the Results page, built from benchmarks/results/*.csv each time the docs are built, so they can't
# drift from the stored runs. Each function returns Markdown, for an `@eval` block.
module ResultsTables

using DelimitedFiles: readdlm
using Markdown: Markdown
using Printf: @sprintf

const RESULTS = joinpath(@__DIR__, "..", "benchmarks", "results")

# Backend keys in the tables, in page order: the name shown and the label.
const BACKENDS = [
    "synchrony" => ("Synchrony", "stable"),
    "hopfield" => ("Hopfield", "stable"),
    "ctrnn" => ("CTRNN design", "stable"),
    "horchler" => ("Horchler", "stable"),
    "synchrony_race" => ("Synchrony, race seed", "experimental"),
    "cylinder" => ("Cylinder", "experimental"),
    "simplex" => ("Simplex", "experimental"),
    "twopop" => ("Two-population", "experimental"),
]

"""The rows of a results table as Dicts by column name."""
function rows(file)
    data, header = readdlm(joinpath(RESULTS, file), ',', Any; header=true)
    return [Dict(String(h) => v for (h, v) in zip(vec(header), r)) for r in eachrow(data)]
end

yes(v) = v == true || v == "true"
mark(v) = yes(v) ? "pass" : "**fail**"
table(header, lines) = Markdown.parse(join([join(header, " | "); join(fill(":--", length(header)), " | "); lines], "\n"))
passed(rs) = "$(count(r -> yes(r["pass"]), rs))/$(length(rs))"
minutes(rs, col) = @sprintf("%.0f", sum(r[col] for r in rs) / 60)

"""One row per backend: label, cases passed in all and at each noise level, and the time the runs took."""
function summary()
    cases = rows("cases.csv")
    lines = map(BACKENDS) do (key, (name, label))
        rs = filter(r -> r["backend"] == key, cases)
        at(ε) = (s = filter(r -> r["noise"] == ε, rs); isempty(s) ? "–" : passed(s))
        join([name, label, passed(rs), at(0.001), at(0.1), minutes(rs, "fit_seconds"), minutes(rs, "check_seconds")], " | ")
    end
    return table(["Backend", "Label", "Passed", "At noise 0.001", "At noise 0.1", "Fits (min)", "Checks (min)"], lines)
end

"""The status table for the home page: backend, label and cases passed."""
function status()
    cases = rows("cases.csv")
    lines = [join([name, label, passed(filter(r -> r["backend"] == key, cases))], " | ") for (key, (name, label)) in BACKENDS]
    return table(["Backend", "Label", "Benchmark cases passed"], lines)
end

"""Every case of one backend: the three checks, the worst dwell error and off-graph share on set 1, and the class."""
function cases(key)
    rs = filter(r -> r["backend"] == key, rows("cases.csv"))
    lines = map(rs) do r
        join([r["case"], r["noise"], mark(r["pass"]), mark(r["set1_pass"]), mark(r["set2_pass"]), mark(r["half_pass"]),
              @sprintf("%.1f", 100r["worst_err"]), @sprintf("%.2f", 100r["off"]), r["class"],
              @sprintf("%.0f", r["fit_seconds"]), @sprintf("%.0f", r["check_seconds"])], " | ")
    end
    return table(["Network", "Noise", "Accepted", "Set 1", "Set 2", "Half step", "Worst dwell error (%)", "Off graph (%)",
                  "Class", "Fit (s)", "Checks (s)"], lines)
end

const CHECKS = ["set1" => ("set1_pass", "Set 1"), "set2" => ("set2_pass", "Set 2"), "half_step" => ("half_pass", "Half step")]

"""What each failed check of one backend missed: branches outside their interval and dwells outside the 5% band."""
function misses(key)
    same(r, x) = x["backend"] == key && x["case"] == r["case"] && x["noise"] == r["noise"]
    edges, states = rows("edges.csv"), rows("states.csv")
    lines = String[]
    for r in rows("cases.csv")
        r["backend"] == key && !yes(r["pass"]) || continue
        for (col, (passcol, label)) in CHECKS
            yes(r[passcol]) && continue
            what = [@sprintf("branch %d → %d: %.3f for %.3g", e["from"], e["to"], e[col], e["designed"])
                    for e in edges if same(r, e) && !yes(e[col * "_ok"])]
            append!(what, [@sprintf("dwell of state %d: %.3g for %.3g", s["state"], s[col], s["target"])
                           for s in states if same(r, s) && abs(s[col] / s["target"] - 1) > 0.05])
            push!(lines, join([r["case"], r["noise"], label, isempty(what) ? "off-graph share or a failed run" : join(what, "; ")], " | "))
        end
    end
    return table(["Network", "Noise", "Check", "What missed"], lines)
end

"""Per backend: how many cases failed, how many checks each of those failed, and the size of the misses."""
function failures()
    cases, edges, states = rows("cases.csv"), rows("edges.csv"), rows("states.csv")
    lines = map(BACKENDS) do (key, (name, _))
        f = filter(r -> r["backend"] == key && !yes(r["pass"]), cases)
        nfail = [count(c -> !yes(r[c[2][1]]), CHECKS) for r in f]
        miss = [abs(e[col] - e["designed"]) for e in edges if e["backend"] == key for (col, _) in CHECKS if !yes(e[col * "_ok"])]
        # a dwell outside the band on any of the three checks (cases.csv has the set 1 error only)
        off_band = unique((s["case"], s["noise"]) for s in states if s["backend"] == key
                          for (col, _) in CHECKS if abs(s[col] / s["target"] - 1) > 0.05)
        join([name, length(f), count(==(1), nfail), count(==(2), nfail), count(==(3), nfail), length(off_band),
              isempty(miss) ? "–" : @sprintf("%.1f to %.1f", 100minimum(miss), 100maximum(miss))], " | ")
    end
    return table(["Backend", "Cases failed", "Failing one check", "Two", "Three", "With a dwell outside 5%",
                  "Branch misses (percentage points)"], lines)
end

"""The reproduction subset: whether each refit agrees with the prototype's stored result."""
function reproduction()
    name = Dict(k => v[1] for (k, v) in BACKENDS)
    lines = map(rows("reproduction.csv")) do r
        join([name[r["backend"]], r["case"], r["noise"], r["expect"] == "exact" ? "equal numbers" : "same band verdicts",
              yes(r["agree"]) ? "yes" : "**no**", mark(r["stored_pass"]), mark(r["new_pass"])], " | ")
    end
    return table(["Backend", "Network", "Noise", "Compared by", "Agrees", "Prototype", "Package"], lines)
end

"""The commit, thread count and dates of the stored suite runs, as a sentence."""
function provenance()
    cases = rows("cases.csv")
    commits, threads = unique(string(r["commit"]) for r in cases), unique(Int(r["threads"]) for r in cases)
    days = sort(unique(first(string(r["date"]), 10) for r in cases))
    return Markdown.parse("The stored runs are $(length(cases)) cases, made at commit $(join(("`$c`" for c in commits), ", ")) on " *
                          "$(join(threads, ", ")) threads, on $(join(days, " and ")).")
end

end
