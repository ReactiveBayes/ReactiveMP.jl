# The documentation site of AutoregressiveMessagePassingRules. Build it with
# `make docs-autoregressive` from the repository root, or `julia --project=docs docs/make.jl` here.
using Documenter, DocumenterInterLinks
using AutoregressiveMessagePassingRules

# Another package's site, for `@extref` links: its planned address, and the inventory of its local
# build, which must exist, so sites build in dependency order (`make docs-all`).
sibling(name) = (
    "https://reactivebayes.github.io/$(name).jl/dev/",
    joinpath(@__DIR__, "..", "..", name, "docs", "build", "objects.inv"),
)

links = InterLinks(
    "MessagePassingRulesBase" => sibling("MessagePassingRulesBase"),
    "StandardMessagePassingRules" => sibling("StandardMessagePassingRules"),
)

# The doctests call rules on distributions, so they load the rule interface and the distributions.
DocMeta.setdocmeta!(
    AutoregressiveMessagePassingRules,
    :DocTestSetup,
    :(using AutoregressiveMessagePassingRules, MessagePassingRulesBase, BayesBase, ExponentialFamily);
    recursive = true,
)

makedocs(
    modules = [AutoregressiveMessagePassingRules],
    sitename = "AutoregressiveMessagePassingRules.jl",
    # Every docstring appears on a page.
    checkdocs = :all,
    plugins = [links],
    pages = ["Overview" => "index.md", "AR" => "ar.md", "ConjugateAR" => "conjugate-ar.md"],
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
)
