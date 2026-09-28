# Compares posteriors dumped by bench_models.jl: julia compare_posteriors.jl <dir> <variantA> <variantB> <tag> <models...>
# Prints, per model, whether the dumps are identical (==, bit for bit) and the largest difference otherwise.
using Serialization, LinearAlgebra
dir, a, b, tag = ARGS[1:4]
maxdiff(x, y) = x == y ? 0.0 : (x isa Number && y isa Number) ? abs(x - y) :
    (x isa AbstractArray && y isa AbstractArray && size(x) == size(y)) ? maximum(map(maxdiff, x, y); init = 0.0) :
    (x isa Tuple && y isa Tuple && length(x) == length(y)) ? maximum(map(maxdiff, x, y); init = 0.0) :
    (x isa AbstractDict && y isa AbstractDict && keys(x) == keys(y)) ? maximum((maxdiff(x[k], y[k]) for k in keys(x)); init = 0.0) :
    (x === nothing && y === nothing) ? 0.0 : (typeof(x) == typeof(y) && x isa Symbol) ? (x == y ? 0.0 : Inf) : Inf
ok = true
for m in ARGS[5:end]
    pa = deserialize(joinpath(dir, "$(a)_$(m)_$(tag).jls")); pb = deserialize(joinpath(dir, "$(b)_$(m)_$(tag).jls"))
    same = isequal(pa, pb)
    global ok &= same
    println(rpad(m, 12), same ? "identical" : "DIFFERENT, max |Δ| = $(maxdiff(pa, pb))")
end
exit(ok ? 0 : 1)
