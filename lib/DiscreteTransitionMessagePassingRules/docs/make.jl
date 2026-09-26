# The documentation site of DiscreteTransitionMessagePassingRules. Build it with `make docs-discrete-transition` from the repository
# root, which builds the sites it links to first, or `julia --project=docs docs/make.jl` here.
using Documenter, DocumenterInterLinks
using DiscreteTransitionMessagePassingRules

# Another package's site, for `@extref` links: its planned address, and the inventory of its local
# build, which must exist, so sites build in dependency order (`make docs-all`).
sibling(name) = (
    "https://reactivebayes.github.io/$(name).jl/dev/",
    joinpath(@__DIR__, "..", "..", name, "docs", "build", "objects.inv"),
)

links = InterLinks(
    "MessagePassingRulesBase" => sibling("MessagePassingRulesBase"),
)

DocMeta.setdocmeta!(DiscreteTransitionMessagePassingRules, :DocTestSetup, :(using DiscreteTransitionMessagePassingRules); recursive = true)

makedocs(
    modules = [DiscreteTransitionMessagePassingRules],
    sitename = "DiscreteTransitionMessagePassingRules.jl",
    # Every docstring appears on a page.
    checkdocs = :all,
    plugins = [links],
    pages = ["Overview" => "index.md", "Internals" => "internals.md"],
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
)
