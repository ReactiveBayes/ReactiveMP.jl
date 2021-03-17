using Documenter, ReactiveMP

makedocs(
    modules  = [ ReactiveMP ],
    clean    = true,
    sitename = "ReactiveMP.jl",
    pages    = [
        "Home"            => "index.md",
        "Getting started" => "getting-started.md",
        "Extending"       => "extending.md",
        "Distributions"   => "distributions.md",
        "API"             => [
            "Node" => "api/node.md"
        ]
        # "TODO"         => "todo.md",
        # "Contributing" => "contributing.md",
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
