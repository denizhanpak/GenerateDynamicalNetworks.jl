# Opens a notebook in real Pluto, runs it, moves every control through its options, and reports
# any cell that errors. Run with the default environment, which has Pluto:
#   julia notebooks/tools/plutocheck.jl notebooks/phase3_optimize.jl 'name=1,2,3' 'other=1,2'
# Each control is listed with the values to send. PlutoUI's Select sends "puiselect-i", so a
# value written as an integer here is sent in that form.
import Pluto

path = abspath(ARGS[1])
controls = [let (k, v) = split(a, "="); Symbol(k) => parse.(Int, split(v, ",")) end for a in ARGS[2:end]]

session = Pluto.ServerSession()
session.options.server.disable_writing_notebook_files = true
session.options.compiler.threads = "7"
notebook = Pluto.SessionActions.open(session, path; run_async=false)

function report(label)
    bad = [c for c in notebook.cells if c.errored]
    println(label, ": ", length(notebook.cells), " cells, ", length(bad), " errored")
    for c in bad
        println("  CELL: ", first(replace(c.code, "\n" => " "), 90))
        println("  ERROR: ", first(replace(string(get(c.output.body, :msg, c.output.body)), "\n" => " "), 400))
    end
    return length(bad)
end

failures = report("first run")
for (name, values) in controls, v in values
    notebook.bonds[name] = Dict("value" => "puiselect-$v")
    Pluto.set_bond_values_reactive(; session, notebook, bound_sym_names=[name], is_first_values=[false], run_async=false)
    global failures += report("$name = option $v")
end
Pluto.SessionActions.shutdown(session, notebook; async=false)
println(failures == 0 ? "ALL CLEAN" : "FAILURES: $failures")
exit(failures == 0 ? 0 : 1)
