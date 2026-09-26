# The documentation site of BIFMMessagePassingRules. Build it with `make docs-bifm` from the repository
# root, which builds the sites it links to first, or `julia --project=docs docs/make.jl` here.
using Documenter, DocumenterInterLinks
using BIFMMessagePassingRules

# Another package's site, for `@extref` links: its planned address, and the inventory of its local
# build, which must exist, so sites build in dependency order (`make docs-all`).
sibling(name) = (
    "https://reactivebayes.github.io/$(name).jl/dev/",
    joinpath(@__DIR__, "..", "..", name, "docs", "build", "objects.inv"),
)

links = InterLinks(
    "MessagePassingRulesBase" => sibling("MessagePassingRulesBase"),
)

DocMeta.setdocmeta!(BIFMMessagePassingRules, :DocTestSetup, :(using BIFMMessagePassingRules); recursive = true)

makedocs(
    modules = [BIFMMessagePassingRules],
    sitename = "BIFMMessagePassingRules.jl",
    # Every docstring appears on a page.
    checkdocs = :all,
    plugins = [links],
    pages = ["Overview" => "index.md", "BIFM" => "bifm.md", "BIFMHelper" => "bifm-helper.md"],
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
)
