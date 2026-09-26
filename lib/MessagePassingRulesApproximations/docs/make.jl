# The documentation site of MessagePassingRulesApproximations. Build it with
# `make docs-approximations` from the repository root, or `julia --project=docs docs/make.jl` here.
using Documenter, DocumenterInterLinks
using MessagePassingRulesApproximations

# Another package's site, for `@extref` links: its planned address, and the inventory of its local
# build, which must exist, so sites build in dependency order (`make docs-all`).
sibling(name) = (
    "https://reactivebayes.github.io/$(name).jl/dev/",
    joinpath(@__DIR__, "..", "..", name, "docs", "build", "objects.inv"),
)

# This package depends on no sibling, so it links to none.
links = InterLinks()

DocMeta.setdocmeta!(
    MessagePassingRulesApproximations, :DocTestSetup, :(using MessagePassingRulesApproximations); recursive = true
)

makedocs(
    modules = [MessagePassingRulesApproximations],
    sitename = "MessagePassingRulesApproximations.jl",
    # Every docstring appears on a page.
    checkdocs = :all,
    plugins = [links],
    pages = [
        "Overview" => "index.md",
        "Unscented transform" => "unscented.md",
        "Linearization" => "linearization.md",
        "Gauss–Hermite cubature" => "gauss-hermite.md",
        "Smoothing" => "smoothing.md",
    ],
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
)
