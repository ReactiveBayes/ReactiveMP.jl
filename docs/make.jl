using Documenter, ReactiveMP

## https://discourse.julialang.org/t/generation-of-documentation-fails-qt-qpa-xcb-could-not-connect-to-display/60988
## https://gr-framework.org/workstations.html#no-output
ENV["GKSwstype"] = "100"

DocMeta.setdocmeta!(ReactiveMP, :DocTestSetup, :(using ReactiveMP, Distributions); recursive=true)

makedocs(
    modules  = [ ReactiveMP ],
    clean    = true,
    sitename = "ReactiveMP.jl",
    pages    = [
        "Introduction"    => "index.md",
        "User guide" => [ 
            "Getting Started"           => "man/getting-started.md",
            "Model Specification"       => "man/model-specification.md",
            "Constraints Specification" => "man/constraints-specification.md",
            "Meta Specification"        => "man/meta-specification.md",
            "Inference execution"       => "man/inference-execution.md",
            "Advanced Tutorial"         => "man/advanced-tutorial.md",
        ],
        "Custom functionality" => [
            "Custom functional form" => "custom/custom-functional-form.md",
        ],
        "Library" => [
            "Messages"            => "lib/message.md",
            "Functional forms"    => "lib/form.md",
            "Prod implementation" => "lib/prod.md",
            "Factor nodes" => [ 
                "Overview" => "lib/node.md",
                "Flow"     => "lib/nodes/flow.md"
            ],
            "Math utils"       => "lib/math.md",
            "Helper utils"     => "lib/helpers.md"
        ],
        "Examples" => [
            "Overview"                         => "examples/overview.md",
            "Linear Regression"                => "examples/linear_regression.md",
            "Linear Gaussian Dynamical System" => "examples/linear_gaussian_state_space_model.md",
            "Hidden Markov Model"              => "examples/hidden_markov_model.md",
            "Hierarchical Gaussian Filter"     => "examples/hierarchical_gaussian_filter.md",
            "Autoregressive Model"             => "examples/autoregressive.md",
            "Normalizing Flows Tutorial"       => "examples/flow_tutorial.md",
            "Univariate Normal Mixture"        => "examples/univariate_normal_mixture.md",
            "Multivariate Normal Mixture"      => "examples/multivariate_normal_mixture.md",
            "Gamma Mixture"                    => "examples/gamma_mixture.md",
            "Custom Nonlinear Node"            => "examples/custom_nonlinear_node.md",
            "Missing data"                     => "examples/missing_data.md"
        ],
        "Contributing" => "extra/contributing.md",
    ],
    format   = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    )
)

if get(ENV, "CI", nothing) == "true"
    deploydocs(
        repo = "github.com/biaslab/ReactiveMP.jl.git"
    )
end
