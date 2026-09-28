# julia mkv6.jl <env-dir>: the v6 environment, RxInfer 5.5.2 over ReactiveMP 6.5.0 on the same
# Rocket 1.10.0 and GraphPPL 4.8.0 as the v7 variants, from the registry.
using Pkg
Pkg.activate(ARGS[1])
Pkg.add(
    [
        PackageSpec(name = "RxInfer", version = "5.5.2"),
        PackageSpec(name = "ReactiveMP", version = "6.5.0"),
        PackageSpec(name = "Rocket", version = "1.10.0"),
        PackageSpec(name = "GraphPPL", version = "4.8.0"),
    ]
)
Pkg.add(["BenchmarkTools", "StableRNGs", "ExponentialFamily", "BayesBase", "Distributions", "Serialization", "Statistics", "LinearAlgebra", "Random", "Profile"])
Pkg.pin(["RxInfer", "ReactiveMP", "Rocket", "GraphPPL"])
Pkg.precompile()
Pkg.status()
