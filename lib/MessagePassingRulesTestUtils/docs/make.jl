# The documentation site of MessagePassingRulesTestUtils. Build it with `make docs-testutils` from the
# repository root, or `julia --project=docs docs/make.jl` here.
using Documenter, DocumenterInterLinks, DocInventories
using MessagePassingRulesTestUtils

# A registered package's site, by its published inventory. A first request to GitHub Pages can take
# seconds, longer than DocInventories' default timeout of one second, so it waits longer.
registered(name) = Inventory(
    "https://reactivebayes.github.io/$(name).jl/stable/objects.inv";
    root_url = "https://reactivebayes.github.io/$(name).jl/stable/", timeout = 30,
)

links = InterLinks(
    "MessagePassingRulesBase" => registered("MessagePassingRulesBase"),
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
    # Pages that show rule results as cards are larger than the defaults allow.
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        example_size_threshold = 400 * 1024, size_threshold_warn = 400 * 1024, size_threshold = 400 * 1024,
    ),
)
