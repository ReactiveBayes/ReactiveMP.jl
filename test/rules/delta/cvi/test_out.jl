module RulesDeltaCVIOutTest

using Test
using ReactiveMP
using Random
using Distributions
using StableRNGs

g(x) = x
struct EmptyOptimizer end

# test this set with $ make test testset='rules:gamma_inverse:out'
@testset "rules:Delta:cvi:out" begin
    seed = 123
    rng = MersenneTwister(seed)
    output = @call_rule DeltaFn{g}(:out, Marginalisation) (q_ins = FactorProduct((NormalMeanVariance(0, tiny),)), meta = CVIApproximation(rng, 1000, 1, EmptyOptimizer))
    @test mean(output.dist) ≈ 2.5797499734233424e-8
end # testset
end # module
