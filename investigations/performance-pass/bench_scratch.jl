# A rule that declares scratch, called through the engine's MessageMapping: BIFM towards `zprev`.
#   julia --project=<variant>/env bench_scratch.jl <variant> <outfile>
using ReactiveMP, BayesBase, ExponentialFamily, LinearAlgebra, BenchmarkTools, BIFMMessagePassingRules, MessagePassingRulesBase
import MessagePassingRulesBase: Target
import ReactiveMP: MessageMapping
const V, OUT = ARGS[1], get(ARGS, 2, "/dev/null")
msg(d) = Message(d, false, false)
for (dz, du, dy) in ((2, 1, 1), (4, 2, 2), (8, 4, 4))
    A = Matrix(0.9I, dz, dz) + 0.01 * ones(dz, dz); B = ones(dz, du) / du; C = ones(dy, dz) / dz
    algo = BIFMSmoother(A, B, C)
    wmp(d) = MvNormalWeightedMeanPrecision(collect(1.0:d), Matrix((2.0 + d)I, d, d))
    ms = (msg(wmp(dy)), msg(wmp(du)), msg(wmp(dz)))
    mapping = MessageMapping(BIFM, Target{:zprev}(), Val((:out, :in, :znext)), nothing, algo, nothing, nothing, nothing)
    mapping(ms, nothing)
    b = @benchmark $mapping($ms, nothing)
    t = minimum(b); m = median(b)
    line = join((V, "bifm zprev dz=$dz", round(t.time; digits = 1), round(m.time; digits = 1), t.allocs, t.memory), '\t')
    println(line); open(io -> println(io, line), OUT, "a")
end
