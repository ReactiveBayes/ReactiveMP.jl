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

# test this set with $ make test testset='rules:gamma_inverse:out'
@testset "rules:Delta:cvi:out" begin
    seed = 123
    rng = MersenneTwister(seed)
    test_meta = CVIApproximation(rng, 100, 1, EmptyOptimizer)

    sample_list_output = @call_rule DeltaFn{id}(:out, Marginalisation) (
        q_ins = FactorProduct((PointMass(0),)),
        meta = test_meta
    )
    @test mean(sample_list_output.dist) ≈ 0

    sample_list_output = @call_rule DeltaFn{id}(:out, Marginalisation) (
        q_ins = FactorProduct((PointMass([0, 0]),)),
        meta = test_meta
    )
    @test mean(sample_list_output.dist) ≈ [0, 0]

    sample_list_output = @call_rule DeltaFn{second_argument}(:out, Marginalisation) (
        q_ins = FactorProduct((PointMass(0), PointMass(1))),
        meta = test_meta
    )
    @test mean(sample_list_output.dist) ≈ 1

    sample_list_output = @call_rule DeltaFn{second_argument}(:out, Marginalisation) (
        q_ins = FactorProduct((PointMass([0, 0]), PointMass(1))),
        meta = test_meta
    )
    @test mean(sample_list_output.dist) ≈ 1

    sample_list_output = @call_rule DeltaFn{first_argument}(:out, Marginalisation) (
        q_ins = FactorProduct((PointMass([0, 0]), PointMass(1))),
        meta = test_meta
    )
    @test mean(sample_list_output.dist) ≈ [0, 0]

    sample_list_output = @call_rule DeltaFn{second_argument}(:out, Marginalisation) (
        q_ins = FactorProduct((NormalMeanVariance(), PointMass(1))),
        meta = test_meta
    )
    @test mean(sample_list_output.dist) ≈ 1

    sample_list_output = @call_rule DeltaFn{square_sum}(:out, Marginalisation) (
        q_ins = FactorProduct((PointMass(0), PointMass(2))),
        meta = test_meta
    )
    @test mean(sample_list_output.dist) ≈ 4
end # testset
end # module
