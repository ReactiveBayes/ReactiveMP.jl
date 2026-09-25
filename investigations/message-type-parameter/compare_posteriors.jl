# Checks the A and B posteriors (types, means, covariances) and free energies are identical.
#   julia --project=envA compare_posteriors.jl post
using Serialization, RxInfer
dir = ARGS[1]
for f in sort(filter(startswith("A_"), readdir(dir)))
    g = joinpath(dir, "B_" * f[3:end])
    isfile(g) || (println(f, ": no B counterpart"); continue)
    a = deserialize(joinpath(dir, f)); b = deserialize(g)
    println(rpad(f[3:end], 20), isequal(a, b) ? "identical" : "DIFFERENT")
end
