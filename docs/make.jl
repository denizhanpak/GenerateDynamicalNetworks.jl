using GenerateDynamicalNetworks
using Documenter

DocMeta.setdocmeta!(GenerateDynamicalNetworks, :DocTestSetup, :(using GenerateDynamicalNetworks); recursive=true)

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
    pages=[
        "Home" => "index.md",
        "API" => "api.md",
        "Internals" => "internals.md",
    ],
)

deploydocs(;
    repo="github.com/denizhanpak/GenerateDynamicalNetworks.jl",
    devbranch="main",
)
