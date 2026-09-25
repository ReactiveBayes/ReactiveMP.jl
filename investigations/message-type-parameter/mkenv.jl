using Pkg
env, root = ARGS[1], ARGS[2]
Pkg.activate(env)
libs = filter(d -> isdir(joinpath(root, "lib", d)) && d != "MessagePassingRulesTestUtils", readdir(joinpath(root, "lib")))
specs = [PackageSpec(path = root); [PackageSpec(path = joinpath(root, "lib", d)) for d in libs]; PackageSpec(path = expanduser("~/Projects/Julia/RxInfer.jl"))]
Pkg.develop(specs)
Pkg.add(["BenchmarkTools", "StableRNGs", "ExponentialFamily", "BayesBase", "Rocket", "Distributions"])
Pkg.status()
