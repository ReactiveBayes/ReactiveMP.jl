module RulesDeltaCVIOutTest

using Test
using ReactiveMP
using Random
using Distributions
using StableRNGs

id(x) = x
first_argument(x::Real, y::Real) = x
first_argument(x::Real, y::AbstractArray) = x
first_argument(x::AbstractArray, y::Real) = x
first_argument(x::AbstractArray, y::AbstractArray) = x
second_argument(x::Real, y::Real) = y
second_argument(x::AbstractArray, y::Real) = y
second_argument(x::Real, y::AbstractArray) = y
square_sum(x::Real, y::Real) = x^2 + y^2
to_vector(x, y) = [x, y]

struct EmptyOptimizer end

function cvi_out_test(func::Function, factor_product::FactorProduct, meta::CVIApproximation, output, atol::Real = 1e-11)
    sample_list_output = @call_rule DeltaFn{func}(:out, Marginalisation) (
        q_ins = factor_product,
        meta = meta
    )
    @test isapprox(mean(sample_list_output.dist), output, atol = atol)
end

# test this set with $ make test testset='rules:gamma_inverse:out'
@testset "rules:Delta:cvi:out" begin
    seed = 123
    rng = MersenneTwister(seed)
    test_meta = CVIApproximation(rng, 1000, 1, EmptyOptimizer())

    @testset "Exact value comparison (Pointmass)" begin
        for i in 1:100
            cvi_out_test(id, FactorProduct((PointMass(i),)), test_meta, i)
            cvi_out_test(id, FactorProduct((PointMass([i, i]),)), test_meta, [i, i])
            cvi_out_test(first_argument, FactorProduct((PointMass([i, i]), PointMass(i + 1))), test_meta, [i, i])
            cvi_out_test(first_argument, FactorProduct((PointMass(i), PointMass(i + 1))), test_meta, i)
            cvi_out_test(second_argument, FactorProduct((PointMass(i), PointMass(i + 1))), test_meta, i + 1)
            cvi_out_test(second_argument, FactorProduct((PointMass([i, i]), PointMass(i + 1))), test_meta, i + 1)
        end
    end

    @testset "Multivariate normal distributions sampling" begin
        factor_product = FactorProduct((MvNormalMeanPrecision(zeros(2)), PointMass([0, 0])))
        sample_list_output = @call_rule DeltaFn{first_argument}(:out, Marginalisation) (
            q_ins = factor_product,
            meta = test_meta
        )
        @test length(mean(sample_list_output)) === 2
        cvi_out_test(second_argument,
            FactorProduct((MvNormalMeanPrecision(zeros(2)), PointMass(1))),
            test_meta, 1)
    end

    @testset "Bernoulli" begin
        for i in 1:100
            cvi_out_test(first_argument, FactorProduct((PointMass(i), Bernoulli(0.5))), test_meta, i)
            cvi_out_test(second_argument, FactorProduct((PointMass(i), Bernoulli(0.5))), test_meta, 1 / 2, 0.1)
            cvi_out_test(to_vector, FactorProduct(((PointMass(i)), Bernoulli(i / 100.0))), test_meta, [i, i / 100], 0.1)
        end
    end

    @testset "Univariate Normal" begin
        for i in 1:100
            cvi_out_test(first_argument, FactorProduct((PointMass(i), NormalMeanVariance())), test_meta, i, 0.1)
            cvi_out_test(second_argument, FactorProduct((PointMass(i), NormalMeanVariance(-2.0))), test_meta, -2, 0.5)
            cvi_out_test(square_sum, FactorProduct((NormalMeanVariance(i, 1), NormalMeanVariance(0, 1))), test_meta, i^2 + 1.0, 20)
            cvi_out_test(to_vector, FactorProduct((PointMass(i), NormalMeanVariance(-2.0))), test_meta, [i, -2.0], 0.1)
            cvi_out_test(.+, FactorProduct((PointMass(i), NormalMeanVariance(i + 2))), test_meta, 2 * i + 2, 2)
            cvi_out_test(.+, FactorProduct((NormalMeanVariance(-7), PointMass(i))), test_meta, -7 + i, 0.6)
        end
    end

    @testset "Gamma shape rate" begin
        cvi_out_test(first_argument, FactorProduct((PointMass(0), GammaShapeRate())), test_meta, 0)
        cvi_out_test(second_argument, FactorProduct((PointMass(0), GammaShapeRate(2.0, 3.0))), test_meta, 2 / 3, 0.1)
        cvi_out_test(to_vector, FactorProduct((PointMass(2), GammaShapeRate(2.0, 3.0))), test_meta, [2.0, 2 / 3], 0.1)
    end
end # testset
end # module
