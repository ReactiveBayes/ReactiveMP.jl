# The documentation site of GaussianCouplingMessagePassingRules. Build it with `make docs-gaussian-coupling` from the repository
# root, which builds the sites it links to first, or `julia --project=docs docs/make.jl` here.
using Documenter, DocumenterInterLinks
using GaussianCouplingMessagePassingRules

# Another package's site, for `@extref` links: its planned address, and the inventory of its local
# build, which must exist, so sites build in dependency order (`make docs-all`).
sibling(name) = (
    "https://reactivebayes.github.io/$(name).jl/dev/",
    joinpath(@__DIR__, "..", "..", name, "docs", "build", "objects.inv"),
)

links = InterLinks(
    "MessagePassingRulesBase" => sibling("MessagePassingRulesBase"),
)

DocMeta.setdocmeta!(GaussianCouplingMessagePassingRules, :DocTestSetup, :(using GaussianCouplingMessagePassingRules, MessagePassingRulesBase, ExponentialFamily, BayesBase); recursive = true)

makedocs(
    modules = [GaussianCouplingMessagePassingRules],
    sitename = "GaussianCouplingMessagePassingRules.jl",
    # Every docstring appears on a page.
    checkdocs = :all,
    plugins = [links],
    pages = ["Overview" => "index.md"],
    # Pages that show rule results as cards are larger than the defaults allow.
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        example_size_threshold = 400 * 1024, size_threshold_warn = 400 * 1024, size_threshold = 400 * 1024,
    ),
)
