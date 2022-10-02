module RulesDeltaCVIOutTest

using Test
using ReactiveMP
using Random
using Distributions
using StableRNGs

id(x) = x
first_argument(x::Real, y::Real) = x
first_argument(x::Real, y::Array) = x
first_argument(x::Array, y::Real) = x
first_argument(x::Array, y::Array) = x
second_argument(x::Real, y::Real) = y
second_argument(x::Array, y::Real) = y
second_argument(x::Real, y::Array) = y
square_sum(x::Real, y::Real) = x^2 + y^2

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

    @testset "Exact value comparison (sampling from Pointmass)" begin
        cvi_out_test(id, FactorProduct((PointMass(0),)), test_meta, 0)
        cvi_out_test(id, FactorProduct((PointMass([0, 0]),)), test_meta, [0, 0])
        cvi_out_test(first_argument, FactorProduct((PointMass([0, 0]), PointMass(1))), test_meta, [0, 0])
        cvi_out_test(first_argument, FactorProduct((PointMass(0), PointMass(1))), test_meta, 0)
        cvi_out_test(second_argument, FactorProduct((PointMass(0), PointMass(1))), test_meta, 1)
        cvi_out_test(second_argument, FactorProduct((PointMass([0, 0]), PointMass(1))), test_meta, 1)
        cvi_out_test(square_sum, FactorProduct((PointMass(0), PointMass(2))), test_meta, 4)
        cvi_out_test(square_sum, FactorProduct((PointMass(0), PointMass(2))), test_meta, 4)
    end

    @testset "Multivariate normal distributions sampling" begin
        factor_product = FactorProduct((MvNormalMeanPrecision(zeros(2)), PointMass([0, 0])))
        sample_list_output = @call_rule DeltaFn{first_argument}(:out, Marginalisation) (
            q_ins = factor_product,
            meta = test_meta
        )
        @test length(mean(sample_list_output.dist)) === 2
        cvi_out_test(second_argument,
            FactorProduct((MvNormalMeanPrecision(zeros(2)), PointMass(1))),
            test_meta, 1)
    end

    @testset "Bernoulli" begin
        cvi_out_test(first_argument, FactorProduct((PointMass(0), Bernoulli(0.5))), test_meta, 0)
    end

    @testset "Univariate Normal" begin
        cvi_out_test(first_argument, FactorProduct((PointMass(0), NormalMeanVariance())), test_meta, 0)
    end

    @testset "Gamma shape rate" begin
        cvi_out_test(first_argument, FactorProduct((PointMass(0), GammaShapeRate())), test_meta, 0)
    end
end # testset
end # module
