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
            "Messages" => "lib/message.md"
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
