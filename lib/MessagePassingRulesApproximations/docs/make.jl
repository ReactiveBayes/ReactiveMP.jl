# The documentation site of MessagePassingRulesApproximations. Build it with
# `make docs-approximations` from the repository root, or `julia --project=docs docs/make.jl` here.
using Documenter, DocumenterInterLinks, DocInventories
using MessagePassingRulesApproximations

# A registered package's site, by its published inventory. A first request to GitHub Pages can take
# seconds, longer than DocInventories' default timeout of one second, so it waits longer.
registered(name) = Inventory(
    "https://reactivebayes.github.io/$(name).jl/stable/objects.inv";
    root_url = "https://reactivebayes.github.io/$(name).jl/stable/", timeout = 30,
)

# The package depends on no other, and links to MessagePassingRulesBase's glossary only.
links = InterLinks(
    "MessagePassingRulesBase" => registered("MessagePassingRulesBase"),
)

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
        "Choosing a method" => "choosing.md",
        "Unscented transform" => "unscented.md",
        "Linearization" => "linearization.md",
        "Gauss–Hermite cubature" => "gauss-hermite.md",
        "Smoothing" => "smoothing.md",
    ],
    # Pages that show rule results as cards are larger than the defaults allow.
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        example_size_threshold = 400 * 1024, size_threshold_warn = 400 * 1024, size_threshold = 400 * 1024,
    ),
)
