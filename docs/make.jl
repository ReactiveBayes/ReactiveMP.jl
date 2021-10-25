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
            "Getting Started"     => "man/getting-started.md",
            "Fundamentals"        => "man/fundamentals.md",
            "Model Specification" => "man/model-specification.md",
            "Inference execution" => "man/inference-execution.md"
        ],
        "Examples" => [
            "Overview"                         => "examples/overview.md",
            "Linear Gaussian Dynamical System" => "examples/linear_gaussian_state_space_model.md",
            "Hierarchical Gaussian Filter"     => "examples/hierarchical_gaussian_filter.md",
        ],
        "Library" => [
            "Messages"     => "lib/message.md",
            "Factor nodes" => "lib/node.md",
            "Math utils"   => "lib/math.md",
            "Helper utils" => "lib/helpers.md"
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
