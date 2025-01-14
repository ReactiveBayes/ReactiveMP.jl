using Documenter, ReactiveMP

## https://discourse.julialang.org/t/generation-of-documentation-fails-qt-qpa-xcb-could-not-connect-to-display/60988
## https://gr-framework.org/workstations.html#no-output
ENV["GKSwstype"] = "100"

DocMeta.setdocmeta!(ReactiveMP, :DocTestSetup, :(using ReactiveMP, Distributions, ExponentialFamily, BayesBase); recursive=true)

makedocs(
    modules  = [ ReactiveMP ],
    clean    = true,
    sitename = "ReactiveMP.jl",
    pages    = [
        "Introduction"    => "index.md",
        "Library" => [
            "Factor nodes"         => "lib/nodes.md",
            "Messages"             => "lib/message.md",
            "Marginals"            => "lib/marginal.md",
            "Message update rules" => "lib/rules.md",
            "Helper utils"         => "lib/helpers.md",
            "Algebra utils"        => "lib/algebra.md",
            "Specific factor nodes" => [
                "Delta" => "lib/nodes/delta.md",
                "Flow" => "lib/nodes/flow.md",
                "BIFM" => "lib/nodes/bifm.md",
                "Logical" => "lib/nodes/logical.md",
                "Continuous transition" => "lib/nodes/ctransition.md",
                "Autoregressive" => "lib/nodes/ar.md",
                "BinomialPolya" => "lib/nodes/binomial_polya.md",
            ]
        ],
        "Custom functionality" => [
            "Custom functional form" => "custom/custom-functional-form.md",
            "Custom addons"          => "custom/custom-addons.md"
        ],
        "Extra" => [
            "Contributing"     => "extra/contributing.md",
            "Extensions"       => "extra/extensions.md",
            "Exported methods" => "extra/methods.md"
        ]
        
    ],
    format   = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        example_size_threshold = 200 * 1024,
        size_threshold_warn = 200 * 1024,
    )
)

if get(ENV, "CI", nothing) == "true"
    deploydocs(
        repo = "github.com/ReactiveBayes/ReactiveMP.jl.git",
        devbranch = "main", 
        forcepush = true
    )
end
