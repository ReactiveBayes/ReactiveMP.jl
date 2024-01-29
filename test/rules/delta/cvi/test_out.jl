module RulesDeltaCVIOutTest

using ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions, StableRNGs
import ReactiveMP: @test_rules

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

function cvi_out_test(func::Function, factor_product::FactorizedJoint, meta, expected_mean; atol::Real = 1e-8, rtol = 1e-8)
    sample_list_output = @call_rule DeltaFn{func}(:out, Marginalisation) (q_ins = factor_product, meta = meta)
    @test isapprox(mean(sample_list_output), expected_mean, atol = atol, rtol = rtol)
end

# test this set with $ make test testset='rules:gamma_inverse:out'
@testitem "rules:Delta:cvi:out" begin
    @testset "Exact value comparison (Pointmass)" begin
        for N in (500, 1000), i in 1:10
            test_meta = DeltaMeta(method = CVI(StableRNG(123), N, 1, EmptyOptimizer()))
            cvi_out_test(id, FactorizedJoint((PointMass(i),)), test_meta, i)
            cvi_out_test(id, FactorizedJoint((PointMass([i, i]),)), test_meta, [i, i])
            cvi_out_test(first_argument, FactorizedJoint((PointMass(i), PointMass(i + 1))), test_meta, i)
            cvi_out_test(first_argument, FactorizedJoint((PointMass([i, i]), PointMass(i + 1))), test_meta, [i, i])
            cvi_out_test(second_argument, FactorizedJoint((PointMass(i), PointMass(i + 1))), test_meta, i + 1)
            cvi_out_test(second_argument, FactorizedJoint((PointMass([i, i]), PointMass(i + 1))), test_meta, i + 1)
        end
    end

    @testset "Multivariate normal distributions sampling" begin
        test_meta = DeltaMeta(method = CVI(StableRNG(123), 1000, 1, EmptyOptimizer()))
        # factor_product = FactorizedJoint((MvNormalMeanPrecision(zeros(2)), PointMass([0, 0])))
        # sample_list_output = @call_rule DeltaFn{first_argument}(:out, Marginalisation) (q_ins = factor_product, meta = test_meta)

        cvi_out_test(first_argument, FactorizedJoint((MvNormalMeanPrecision(zeros(2)), PointMass(1))), test_meta, zeros(2), atol = 1e-1)
        cvi_out_test(second_argument, FactorizedJoint((MvNormalMeanPrecision(zeros(2)), PointMass(1))), test_meta, 1.0)
    end

    @testset "Bernoulli" begin
        test_meta = DeltaMeta(method = CVI(StableRNG(123), 1000, 1, EmptyOptimizer()))
        for i in 1:100
            cvi_out_test(first_argument, FactorizedJoint((PointMass(i), Bernoulli(0.5))), test_meta, i)
            cvi_out_test(second_argument, FactorizedJoint((PointMass(i), Bernoulli(0.5))), test_meta, 1 / 2; atol = 1e-1)
            cvi_out_test(to_vector, FactorizedJoint(((PointMass(i)), Bernoulli(i / 100.0))), test_meta, [i, i / 100]; atol = 1e-1)
        end
    end

    @testset "Univariate Normal" begin
        test_meta = DeltaMeta(method = CVI(StableRNG(123), 1000, 1, EmptyOptimizer()))
        for i in 1:5
            cvi_out_test(first_argument, FactorizedJoint((PointMass(i), NormalMeanVariance())), test_meta, i)
            cvi_out_test(second_argument, FactorizedJoint((PointMass(i), NormalMeanVariance(-2.0))), test_meta, -2.0; atol = 3e-1)
            cvi_out_test(square_sum, FactorizedJoint((NormalMeanVariance(i, 1), NormalMeanVariance(0, 1))), test_meta, i^2 + 2.0; atol = 3e-1)
            cvi_out_test(to_vector, FactorizedJoint((PointMass(i), NormalMeanVariance(-2.0))), test_meta, [i, -2.0]; atol = 3e-1)
            cvi_out_test(.+, FactorizedJoint((PointMass(i), NormalMeanVariance(i + 2))), test_meta, 2 * i + 2; atol = 3e-1)
            cvi_out_test(.+, FactorizedJoint((NormalMeanVariance(-7), PointMass(i))), test_meta, -7 + i; atol = 3e-1)
        end
    end

    @testset "Gamma shape rate" begin
        test_meta = DeltaMeta(method = CVI(StableRNG(123), 1000, 1, EmptyOptimizer()))
        for i in 1:100
            cvi_out_test(first_argument, FactorizedJoint((PointMass(i), GammaShapeRate())), test_meta, i)
            cvi_out_test(second_argument, FactorizedJoint((PointMass(0), GammaShapeRate(i, 3.0))), test_meta, i / 3.0; atol = 3e-1)
            cvi_out_test(to_vector, FactorizedJoint((PointMass(2), GammaShapeRate(2.0, i))), test_meta, [2.0, 2.0 / i]; atol = 3e-1)
        end
    end
end # testset
end # module
