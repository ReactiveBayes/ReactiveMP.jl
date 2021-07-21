using Documenter, ReactiveMP

DocMeta.setdocmeta!(ReactiveMP, :DocTestSetup, :(using ReactiveMP, Distributions); recursive=true)

makedocs(
    modules  = [ ReactiveMP ],
    clean    = true,
    sitename = "ReactiveMP.jl",
    pages    = [
        "Introduction"    => "index.md",
        "User guide" => [ 
            "Getting Started" => "man/getting-started.md",
            "Model Specification" => "man/model-specification.md"
        ],
        "Library" => [
            "Message" => "lib/message.md"
        ],
        # "Extending"       => "extending.md",
        # "Distributions"   => "distributions.md",
        # "API"             => [
        #     "Node" => "api/node.md"
        # ],
        # "TODO"         => "todo.md",
        "Contributing" => "extra/contributing.md",
        # "Utils"        => "utils.md"
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
