# Runs the tutorials and writes them, with their output and figures, to docs/src/tutorials. Run it by hand after
# changing a tutorial or the package and commit what it writes: the docs build only renders those pages.
#
#   julia --project=docs/literate -t 7 docs/literate/make.jl [name ...]
#
# With no name, all three: ring, branching, two_way. A run depends only on its seeds, so the numbers don't
# depend on the thread count.
#
# The text of a tutorial describes the output of its run. After regenerating, read each page against its new
# numbers and figures.
using Literate

const OUT = joinpath(@__DIR__, "..", "src", "tutorials")

for name in (isempty(ARGS) ? ["ring", "branching", "two_way"] : ARGS)
    Literate.markdown(joinpath(@__DIR__, name * ".jl"), OUT; execute=true, flavor=Literate.DocumenterFlavor(), credit=false)
end
