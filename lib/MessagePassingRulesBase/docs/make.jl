# The documentation site of MessagePassingRulesBase. Build it with `make docs-base` from the
# repository root, or `julia --project=docs docs/make.jl` here.
using Documenter, DocumenterInterLinks
using MessagePassingRulesBase

# Another package's site, for `@extref` links: its planned address, and the inventory of its local
# build, which must exist, so sites build in dependency order (`make docs-all`).
sibling(name) = (
    "https://reactivebayes.github.io/$(name).jl/dev/",
    joinpath(@__DIR__, "..", "..", name, "docs", "build", "objects.inv"),
)

# This package depends on no sibling, so it links to none.
links = InterLinks()

DocMeta.setdocmeta!(MessagePassingRulesBase, :DocTestSetup, :(using MessagePassingRulesBase); recursive = true)

makedocs(
    modules = [MessagePassingRulesBase],
    sitename = "MessagePassingRulesBase.jl",
    # Every docstring appears on a page.
    checkdocs = :all,
    plugins = [links],
    pages = [
        "Overview" => "index.md",
        "Defining nodes" => "nodes.md",
        "Defining rules" => "rules.md",
        "Algorithms and dependencies" => "algorithms.md",
        "Log scales" => "logscales.md",
        "The rule context" => "context.md",
        "Calling rules" => "calling.md",
        "Inspecting rules" => "inspecting.md",
        "Rule fallbacks" => "fallbacks.md",
        "Internals" => "internals.md",
    ],
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
)
