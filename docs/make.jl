using Documenter
using ReactiveMP, MessagePassingRulesBase, MessagePassingRulesTestUtils
using StandardMessagePassingRules, MessagePassingRulesApproximations, DeltaMessagePassingRules
using GaussianCouplingMessagePassingRules, ProbitMessagePassingRules, GCVMessagePassingRules, SoftDotMessagePassingRules
using AutoregressiveMessagePassingRules, ContinuousTransitionMessagePassingRules, PolyaMessagePassingRules, BIFMMessagePassingRules

const MODULES = [
    ReactiveMP, MessagePassingRulesBase, MessagePassingRulesTestUtils,
    StandardMessagePassingRules, MessagePassingRulesApproximations, DeltaMessagePassingRules,
    GaussianCouplingMessagePassingRules, ProbitMessagePassingRules, GCVMessagePassingRules, SoftDotMessagePassingRules,
    AutoregressiveMessagePassingRules, ContinuousTransitionMessagePassingRules, PolyaMessagePassingRules,
    BIFMMessagePassingRules,
]

const SETUP = :(using ReactiveMP, MessagePassingRulesBase, StandardMessagePassingRules, BayesBase, Distributions, ExponentialFamily)
foreach(m -> DocMeta.setdocmeta!(m, :DocTestSetup, SETUP; recursive = true), MODULES)

makedocs(
    modules = MODULES,
    clean = true,
    sitename = "ReactiveMP.jl",
    # Every docstring appears on a page, exported or not; an internal helper has a comment.
    checkdocs = :all,
    pages = [
        "Introduction" => "index.md",
        "Concepts" => [
            "Factor graphs" => "concepts/factor-graphs.md",
            "Message passing" => "concepts/message-passing.md",
            "Reactive programming" => "concepts/reactive-programming.md",
            "Inference lifecycle" => "concepts/inference-lifecycle.md",
        ],
        "Nodes and rules" => [
            "Defining nodes and rules" => "rules/defining-nodes-and-rules.md",
            "Algorithms and dependencies" => "rules/algorithms-and-dependencies.md",
            "Testing rules" => "rules/testing-rules.md",
        ],
        "The engine" => [
            "Factor nodes" => "lib/nodes.md",
            "Variables" => "lib/variables.md",
            "Messages" => "lib/message.md",
            "Marginals" => "lib/marginal.md",
            "Callbacks" => "lib/callbacks.md",
            "Stream postprocessors" => "lib/stream-postprocessors.md",
            "Free energy" => "lib/score.md",
            "Form constraints" => "custom/custom-functional-form.md",
            "Helpers" => "lib/helpers.md",
            "Annotations" => [
                "Overview" => "lib/annotations.md",
                "Log scale" => "lib/annotations/logscale.md",
                "Input arguments" => "lib/annotations/input_arguments.md",
            ],
        ],
        "Rule packages" => [
            "Standard rules" => "packages/standard.md",
            "Approximations" => "packages/approximations.md",
            "The Delta node" => "packages/delta.md",
            "GaussianCoupling" => "packages/gaussian-coupling.md",
            "Probit" => "packages/probit.md",
            "GCV" => "packages/gcv.md",
            "SoftDot" => "packages/softdot.md",
            "Autoregressive" => "packages/autoregressive.md",
            "ContinuousTransition" => "packages/continuous-transition.md",
            "Pólya" => "packages/polya.md",
            "BIFM" => "packages/bifm.md",
        ],
        "Migration guides" => [
            "v6 to v7" => "migration-guides/v6-to-v7.md",
            "v5 to v6" => "migration-guides/v5-to-v6.md",
        ],
        "Extra" => [
            "Contributing" => "extra/contributing.md",
            "Exported methods" => "extra/methods.md",
        ],
    ],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        example_size_threshold = 400 * 1024,
        size_threshold_warn = 400 * 1024,
        size_threshold = 400 * 1024,
    ),
)

if get(ENV, "CI", nothing) == "true"
    deploydocs(repo = "github.com/ReactiveBayes/ReactiveMP.jl.git", devbranch = "main", forcepush = true)
end
