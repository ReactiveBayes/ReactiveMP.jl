# The documentation site of ReactiveMP, the engine. Build it with `make docs` from the repository
# root, after the package sites it links to (`make docs-all` builds them all, this one last).
using Documenter, DocumenterInterLinks, DocInventories
using ReactiveMP

# Another package's site, for `@extref` links: its planned address, and the inventory of its local
# build, which must exist, so sites build in dependency order (`make docs-all`).
sibling(name) = (
    "https://reactivebayes.github.io/$(name).jl/dev/",
    joinpath(@__DIR__, "..", "lib", name, "docs", "build", "objects.inv"),
)

const SIBLINGS = [
    "MessagePassingRulesTestUtils",
    "StandardMessagePassingRules",
    "MessagePassingRulesApproximations",
    "DeltaMessagePassingRules",
    "GaussianCouplingMessagePassingRules",
    "ProbitMessagePassingRules",
    "GCVMessagePassingRules",
    "SoftDotMessagePassingRules",
    "AutoregressiveMessagePassingRules",
    "ContinuousTransitionMessagePassingRules",
    "PolyaMessagePassingRules",
    "BIFMMessagePassingRules",
    "FlowMessagePassingRules",
    "DiscreteTransitionMessagePassingRules",
]

# A registered package's site, by its published inventory. A first request to GitHub Pages can take
# seconds, longer than DocInventories' default timeout of one second, so it waits longer.
registered(name) = Inventory(
    "https://reactivebayes.github.io/$(name).jl/stable/objects.inv";
    root_url = "https://reactivebayes.github.io/$(name).jl/stable/", timeout = 30,
)

const REGISTERED = ["MessagePassingRulesBase" => registered("MessagePassingRulesBase")]

links = InterLinks(REGISTERED..., (name => sibling(name) for name in SIBLINGS)...)

DocMeta.setdocmeta!(
    ReactiveMP, :DocTestSetup, :(using ReactiveMP, BayesBase, Distributions, ExponentialFamily); recursive = true
)

makedocs(
    modules = [ReactiveMP],
    sitename = "ReactiveMP.jl",
    # Every docstring appears on a page.
    checkdocs = :all,
    plugins = [links],
    pages = [
        "Introduction" => "index.md",
        "Getting started" => "getting-started.md",
        "The example node" => "example-node.md",
        "Concepts" => [
            "Factor graphs" => "concepts/factor-graphs.md",
            "Message passing" => "concepts/message-passing.md",
            "Inference lifecycle" => "concepts/inference-lifecycle.md",
        ],
        "The engine" => [
            "Variables" => "lib/variables.md",
            "Factor nodes" => "lib/nodes.md",
            "Activation options" => "lib/activation-options.md",
            "Messages" => "lib/message.md",
            "Marginals" => "lib/marginal.md",
            "Log scales" => "lib/logscale.md",
            "Form constraints" => "lib/form-constraints.md",
            "Free energy" => "lib/score.md",
        ],
        "Extension points" => [
            "Callbacks" => "lib/callbacks.md",
            "Stream postprocessors" => "lib/stream-postprocessors.md",
            "Annotations" => "lib/annotations.md",
        ],
        "The ecosystem" => "ecosystem.md",
        "Migration guides" => [
            "v6 to v7" => "migration-guides/v6-to-v7.md",
            "v5 to v6" => "migration-guides/v5-to-v6.md",
        ],
        "Contributing" => "extra/contributing.md",
        "Internals" => "internals.md",
    ],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        edit_link = "main",
        example_size_threshold = 400 * 1024,
        size_threshold_warn = 400 * 1024,
        size_threshold = 400 * 1024,
    ),
)

if get(ENV, "CI", nothing) == "true"
    deploydocs(repo = "github.com/ReactiveBayes/ReactiveMP.jl.git", devbranch = "main", forcepush = true)
end
