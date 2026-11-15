using GenerateDynamicalNetworks
using Documenter
using DocumenterCitations

# The tables on the Results and Home pages, read from benchmarks/results; pages call it as Main.ResultsTables.
include("results_tables.jl")

DocMeta.setdocmeta!(GenerateDynamicalNetworks, :DocTestSetup, :(using GenerateDynamicalNetworks); recursive=true)

# The exported research notebooks, linked from the Research record page: copied into the source tree for the build
# (the copies are ignored by git).
let from = joinpath(@__DIR__, "..", "research", "notebooks", "html"), to = joinpath(@__DIR__, "src", "notebooks")
    mkpath(to)
    for f in filter(endswith(".html"), readdir(from))
        cp(joinpath(from, f), joinpath(to, f); force=true)
    end
end

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"); style=:authoryear)

makedocs(;
    modules=[GenerateDynamicalNetworks],
    authors="denizhanpak <denpak@iu.edu>",
    repo=Documenter.Remotes.GitHub("denizhanpak", "GenerateDynamicalNetworks.jl"),
    sitename="GenerateDynamicalNetworks.jl",
    format=Documenter.HTML(;
        canonical="https://denizhanpak.github.io/GenerateDynamicalNetworks.jl",
        edit_link="main",
        assets=String[],
    ),
    plugins=[bib],
    pages=[
        "Home" => "index.md",
        # The tutorials are written by docs/literate/make.jl, run by hand, and committed; this build only renders them.
        "Tutorials" => [
            "A three-state ring" => "tutorials/ring.md",
            "A branching graph" => "tutorials/branching.md",
            "A two-way graph" => "tutorials/two_way.md",
        ],
        "Concepts" => [
            "The specification" => "concepts/specification.md",
            "Simulation and decoding" => "concepts/simulation.md",
            "The acceptance rule" => "concepts/acceptance.md",
            "Dynamical classes" => "concepts/classes.md",
        ],
        "Backends" => [
            "Horchler" => "backends/horchler.md",
            "Simplex" => "backends/simplex.md",
            "Cylinder" => "backends/cylinder.md",
            "Two-population" => "backends/twopop.md",
            "CTRNN design" => "backends/ctrnn.md",
            "Synchrony" => "backends/synchrony.md",
            "Hopfield" => "backends/hopfield.md",
        ],
        "Results" => "results.md",
        "Reference" => [
            "API" => "api.md",
            "Internals" => "internals.md",
        ],
        "Research record" => "research.md",
        "References" => "references.md",
    ],
)

deploydocs(;
    repo="github.com/denizhanpak/GenerateDynamicalNetworks.jl",
    devbranch="main",
)
