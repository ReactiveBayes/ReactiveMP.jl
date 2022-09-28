module RulesDeltaCVIOutTest

using Test
using ReactiveMP
using Random
using Distributions
using StableRNGs

id(x) = x
first_argument(x, y) = x
second_argument(x, y) = y
square_sum(x, y) = x^2 + y^2

struct EmptyOptimizer end

function cvi_out_test(func::Function, factor_product::FactorProduct, meta::CVIApproximation, output)
    sample_list_output = @call_rule DeltaFn{func}(:out, Marginalisation) (
        q_ins = factor_product,
        meta = meta
    )
    @test mean(sample_list_output.dist) ≈ output
end

# test this set with $ make test testset='rules:gamma_inverse:out'
@testset "rules:Delta:cvi:out" begin
    seed = 123
    rng = MersenneTwister(seed)
    test_meta = CVIApproximation(rng, 100, 1, EmptyOptimizer)

    cvi_out_test(id, FactorProduct((PointMass(0),)), test_meta, 0)
    cvi_out_test(id, FactorProduct((PointMass([0, 0]),)), test_meta, [0, 0])
    cvi_out_test(first_argument, FactorProduct((PointMass([0, 0]), PointMass(1))), test_meta, [0, 0])
    cvi_out_test(first_argument, FactorProduct((PointMass(0), PointMass(1))), test_meta, 0)
    cvi_out_test(second_argument, FactorProduct((PointMass(0), PointMass(1))), test_meta, 1)
    cvi_out_test(second_argument, FactorProduct((PointMass([0, 0]), PointMass(1))), test_meta, 1)
    cvi_out_test(second_argument, FactorProduct((NormalMeanVariance(), PointMass(1))), test_meta, 1)
    cvi_out_test(square_sum, FactorProduct((PointMass(0), PointMass(2))), test_meta, 4)
end # testset
end # module
