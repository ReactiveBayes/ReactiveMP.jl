# Compares posteriors and free energies dumped by bench_models.jl between a reference variant and
# others, for every model and tag the reference has: identical, or the largest absolute difference.
#   julia --project=<any v7 env> compare_variants.jl <dir> <reference-variant> <variant>...
using Serialization, LinearAlgebra, RxInfer, ExponentialFamily, Distributions, BayesBase
dir, ref = ARGS[1], ARGS[2]
maxdiff(a::Number, b::Number) = abs(a - b)
maxdiff(a::AbstractArray, b::AbstractArray) = size(a) == size(b) ? maximum(maxdiff.(a, b); init = 0.0) : Inf
maxdiff(a::Tuple, b::Tuple) = length(a) == length(b) ? maximum(map(maxdiff, a, b); init = 0.0) : Inf
maxdiff(a::AbstractDict, b::AbstractDict) = keys(a) == keys(b) ? maximum((maxdiff(a[k], b[k]) for k in keys(a)); init = 0.0) : Inf
maxdiff(a, b) = isequal(a, b) ? 0.0 : Inf
bad = 0
for f in filter(f -> startswith(f, ref * "_") && endswith(f, ".jls"), readdir(dir))
    rest = f[(length(ref) + 2):end]
    a = deserialize(joinpath(dir, f))
    for v in ARGS[3:end]
        g = joinpath(dir, v * "_" * rest)
        isfile(g) || continue
        b = deserialize(g)
        d = isequal(a, b) ? 0.0 : maxdiff(a, b)
        global bad += d > 1.0e-12
        println(rpad(rest, 32), rpad(v, 10), isequal(a, b) ? "identical" : "maxdiff=$d")
    end
end
exit(bad == 0 ? 0 : 1)
