# The documentation site of PolyaMessagePassingRules. Build it with `make docs-polya` from the repository
# root, which builds the sites it links to first, or `julia --project=docs docs/make.jl` here.
using Documenter, DocumenterInterLinks
using PolyaMessagePassingRules

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

DocMeta.setdocmeta!(PolyaMessagePassingRules, :DocTestSetup, :(using PolyaMessagePassingRules); recursive = true)

makedocs(
    modules = [PolyaMessagePassingRules],
    sitename = "PolyaMessagePassingRules.jl",
    # Every docstring appears on a page.
    checkdocs = :all,
    plugins = [links],
    pages = ["Overview" => "index.md", "BinomialPolya" => "binomial.md", "MultinomialPolya" => "multinomial.md"],
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
)
