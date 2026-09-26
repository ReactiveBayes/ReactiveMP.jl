# The documentation site of FlowMessagePassingRules. Build it with `make docs-flow` from the repository
# root, or `julia --project=docs docs/make.jl` here.
using Documenter, DocumenterInterLinks
using FlowMessagePassingRules

# Another package's site, for `@extref` links: its planned address, and the inventory of its local
# build, which must exist, so sites build in dependency order (`make docs-all`).
sibling(name) = (
    "https://reactivebayes.github.io/$(name).jl/dev/",
    joinpath(@__DIR__, "..", "..", name, "docs", "build", "objects.inv"),
)

links = InterLinks(
    "MessagePassingRulesApproximations" => sibling("MessagePassingRulesApproximations"),
    "MessagePassingRulesBase" => sibling("MessagePassingRulesBase"),
)

DocMeta.setdocmeta!(FlowMessagePassingRules, :DocTestSetup, :(using FlowMessagePassingRules); recursive = true)

makedocs(
    modules = [FlowMessagePassingRules],
    sitename = "FlowMessagePassingRules.jl",
    # Every docstring appears on a page.
    checkdocs = :all,
    plugins = [links],
    pages = ["Flow" => "index.md", "Flow models" => "models.md", "Internals" => "internals.md"],
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
)
