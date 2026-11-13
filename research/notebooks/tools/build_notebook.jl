# Builds a Pluto notebook from a plain file of cells separated by lines reading `#---`.
#   julia notebooks/tools/build_notebook.jl notebooks/tools/phase3_cells.jl notebooks/phase3_optimize.jl
using UUIDs

const HEADER = raw"""
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
"""

source, target = ARGS
cells = strip.(split(read(source, String), r"^#---$"m))
filter!(!isempty, cells)
ids = [uuid5(UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8"), "$(basename(target)):$i") for i in eachindex(cells)]
open(target, "w") do io
    println(io, lstrip(HEADER))
    foreach((id, c) -> println(io, "# ╔═╡ $id\n$c\n"), ids, cells)
    println(io, "# ╔═╡ Cell order:")
    foreach(id -> println(io, "# ╠═$id"), ids)
end
println(length(cells), " cells written to ", target)
