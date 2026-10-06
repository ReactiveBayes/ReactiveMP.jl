# The documentation site of StandardMessagePassingRules. Build it with `make docs-standard` from the repository
# root, which builds the sites it links to first, or `julia --project=docs docs/make.jl` here.
using Documenter, DocumenterInterLinks, DocInventories
using StandardMessagePassingRules

# Another package's site, for `@extref` links: its planned address, and the inventory of its local
# build, which must exist, so sites build in dependency order (`make docs-all`).
sibling(name) = (
    "https://reactivebayes.github.io/$(name).jl/dev/",
    joinpath(@__DIR__, "..", "..", name, "docs", "build", "objects.inv"),
)

# A registered package's site, by its published inventory. A first request to GitHub Pages can take
# seconds, longer than DocInventories' default timeout of one second, so it waits longer.
registered(name) = Inventory(
    "https://reactivebayes.github.io/$(name).jl/stable/objects.inv";
    root_url = "https://reactivebayes.github.io/$(name).jl/stable/", timeout = 30,
)

links = InterLinks(
    "MessagePassingRulesBase" => registered("MessagePassingRulesBase"),
)

DocMeta.setdocmeta!(StandardMessagePassingRules, :DocTestSetup, :(using StandardMessagePassingRules); recursive = true)

makedocs(
    modules = [StandardMessagePassingRules],
    sitename = "StandardMessagePassingRules.jl",
    # Every docstring appears on a page.
    checkdocs = :all,
    plugins = [links],
    pages = [
        "Overview" => "index.md",
        "Normal distributions" => "normal.md",
        "Gamma, Beta and discrete distributions" => "gamma-beta-discrete.md",
        "Matrix and joint distributions" => "matrix-joint.md",
        "Arithmetic" => "arithmetic.md",
        "Logic" => "logic.md",
        "Mixtures" => "mixtures.md",
        "Helper nodes and message types" => "helpers.md",
        "Internals" => "internals.md",
    ],
    # Pages that show rule results as cards are larger than the defaults allow.
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        example_size_threshold = 400 * 1024, size_threshold_warn = 400 * 1024, size_threshold = 400 * 1024,
    ),
)
