# Runs notebooks in Pluto and writes each as a static HTML page to notebooks/html/. The notebook files aren't
# changed. Run with the default environment, which has Pluto:
#   julia research/notebooks/tools/export_html.jl research/notebooks/phase0_benchmarks.jl [more notebooks]
# The page shows every cell's output as it was after the run, with controls at their starting values. It loads
# Pluto's scripts and styles from the web when opened.
import Pluto

for path in abspath.(ARGS)
    session = Pluto.ServerSession()
    session.options.server.disable_writing_notebook_files = true
    session.options.compiler.threads = "7"
    seconds = @elapsed notebook = Pluto.SessionActions.open(session, path; run_async=false)
    bad = [c for c in notebook.cells if c.errored]
    out = joinpath(dirname(path), "html", replace(basename(path), r"\.jl$" => ".html"))
    mkpath(dirname(out))
    # Keep this machine's paths out of the page: the notebook's own path, and the cells' log entries (package
    # messages), which name files under the home folder.
    foreach(c -> empty!(c.logs), notebook.cells)
    notebook.path = basename(path)
    write(out, Pluto.generate_html(notebook))
    notebook.path = path
    println(basename(path), ": ", length(notebook.cells), " cells, ", length(bad), " errored, ", round(Int, seconds), " s, ",
            round(filesize(out) / 1e6; digits=2), " MB")
    for c in bad
        println("  CELL: ", first(replace(c.code, "\n" => " "), 90))
        println("  ERROR: ", first(replace(string(get(c.output.body, :msg, c.output.body)), "\n" => " "), 300))
    end
    Pluto.SessionActions.shutdown(session, notebook; async=false)
end
