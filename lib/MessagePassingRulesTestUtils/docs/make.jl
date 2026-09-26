# The documentation site of MessagePassingRulesTestUtils. Build it with `make docs-testutils` from the
# repository root, after `make docs-base`, whose inventory it links to, or `julia --project=docs docs/make.jl`
# here.
using Documenter, DocumenterInterLinks
using MessagePassingRulesTestUtils

# Another package's site, for `@extref` links: its planned address, and the inventory of its local
# build, which must exist, so sites build in dependency order (`make docs-all`).
sibling(name) = (
    "https://reactivebayes.github.io/$(name).jl/dev/",
    joinpath(@__DIR__, "..", "..", name, "docs", "build", "objects.inv"),
)

links = InterLinks(
    "MessagePassingRulesBase" => sibling("MessagePassingRulesBase"),
)

DocMeta.setdocmeta!(MessagePassingRulesTestUtils, :DocTestSetup, :(using MessagePassingRulesTestUtils); recursive = true)

makedocs(
    modules = [MessagePassingRulesTestUtils],
    sitename = "MessagePassingRulesTestUtils.jl",
    # Every docstring appears on a page.
    checkdocs = :all,
    plugins = [links],
    pages = [
        "Overview" => "index.md",
        "Table tests" => "tables.md",
        "Verification against the node" => "verification.md",
        "Derivatives" => "derivatives.md",
        "The rule-coverage gate" => "coverage.md",
        "Comparing with a reference" => "reference.md",
        "Engine trajectories" => "engine.md",
    ],
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
)
