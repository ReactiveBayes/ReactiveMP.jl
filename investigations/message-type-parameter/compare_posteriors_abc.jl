# Checks every listed variant's posteriors and free energies equal A's, bit for bit.
#   julia --project=envA compare_posteriors_abc.jl <dir> B,C,D,E
using Serialization, RxInfer
dir = ARGS[1]
variants = split(ARGS[2], ",")
for f in sort(filter(startswith("A_"), readdir(dir))), v in variants
    g = joinpath(dir, v * f[2:end])
    isfile(g) || (println(v, f[2:end], ": missing"); continue)
    println(rpad(v * " vs A " * f[3:end], 28), isequal(deserialize(joinpath(dir, f)), deserialize(g)) ? "identical" : "DIFFERENT")
end
