# Round 2's correctness gate: the posteriors (every parameter, the type in full) and the free
# energies bench_models2.jl dumped, of each variant against a reference, model by model and round
# by round. Prints the reference it compared against on every line.
#   julia --project=<any v7 env> compare_variants2.jl <post dir> <reference> <variant>...
# Exit code 1 when anything differs by more than 1e-12 (`isequal` first: identical means bitwise).
using Serialization, LinearAlgebra, RxInfer, ExponentialFamily, Distributions, BayesBase
dir, ref = ARGS[1], ARGS[2]
maxdiff(a::Number, b::Number) = abs(a - b)
maxdiff(a::AbstractString, b::AbstractString) = a == b ? 0.0 : Inf
maxdiff(a::AbstractArray, b::AbstractArray) = size(a) == size(b) ? maximum(maxdiff.(a, b); init = 0.0) : Inf
maxdiff(a::Tuple, b::Tuple) = length(a) == length(b) ? maximum(map(maxdiff, a, b); init = 0.0) : Inf
maxdiff(a::AbstractDict, b::AbstractDict) = keys(a) == keys(b) ? maximum((maxdiff(a[k], b[k]) for k in keys(a)); init = 0.0) : Inf
maxdiff(a, b) = isequal(a, b) ? 0.0 : Inf
bad = 0; n = 0
for f in sort(filter(f -> startswith(f, ref * "_") && endswith(f, ".jls"), readdir(dir)))
    rest = f[(length(ref) + 2):end]
    a = deserialize(joinpath(dir, f))
    for v in ARGS[3:end]
        g = joinpath(dir, v * "_" * rest)
        isfile(g) || continue
        b = deserialize(g)
        d = isequal(a, b) ? 0.0 : maxdiff(a, b)
        global bad += d > 1.0e-12; global n += 1
        println(rpad(rest, 34), rpad(v, 10), "vs ", rpad(ref, 8), isequal(a, b) ? "identical" : "maxdiff=$d")
    end
end
println("compared $n, differing $bad")
exit(bad == 0 ? 0 : 1)
