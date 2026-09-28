# julia mkenv.jl <env-dir> <variant-dir>: an environment developing the variant's worktrees.
using Pkg
env, root = ARGS[1], ARGS[2]
rmp = joinpath(root, "ReactiveMP.jl")
Pkg.activate(env)
libs = filter(d -> isdir(joinpath(rmp, "lib", d)) && d != "MessagePassingRulesTestUtils", readdir(joinpath(rmp, "lib")))
specs = [
    PackageSpec(path = rmp);
    [PackageSpec(path = joinpath(rmp, "lib", d)) for d in libs];
    PackageSpec(path = joinpath(root, "RxInfer.jl"));
    PackageSpec(path = joinpath(root, "Rocket.jl"));
    PackageSpec(path = joinpath(root, "GraphPPL.jl"))
]
Pkg.develop(specs)
Pkg.add(["BenchmarkTools", "Chairmarks", "StableRNGs", "ExponentialFamily", "BayesBase", "Distributions", "JET", "Serialization", "Statistics", "LinearAlgebra", "Random", "Profile", "PProf"])
Pkg.precompile()
Pkg.status()
