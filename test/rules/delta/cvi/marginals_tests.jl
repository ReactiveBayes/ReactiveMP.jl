
@testitem "marginalrules:CVI" begin
    using ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions, StableRNGs, Optimisers, LinearAlgebra
    import ReactiveMP: @test_rules, @test_marginalrules

    add_1 = (x::Real) -> x + 1

    function two_into_one(x::Real, y::Real)
        return [x, y]
    end

    function extract_coordinate(x::Vector)
        return x[1]
    end

    @testset "id, x~Normal, y~Normal" begin
        for enforce in (Val(false), Val(true)), grad in (ForwardDiffGrad(), ForwardDiffGrad(1))
            test_meta = DeltaMeta(method = CVI(StableRNG(123), 1, 1000, Optimisers.Descent(0.007), grad, 20, enforce, true))
            @test_marginalrules [check_type_promotion = false, atol = 1e-2] DeltaFn{identity}(:ins) [
                (input = (m_out = NormalMeanVariance(1.0, 1.0), m_ins = ManyOf(NormalMeanVariance()), meta = test_meta), output = FactorizedJoint((NormalMeanVariance(0.5, 0.5),))),
                (input = (m_out = NormalMeanVariance(2.0, 1.0), m_ins = ManyOf(NormalMeanVariance()), meta = test_meta), output = FactorizedJoint((NormalMeanVariance(1.0, 0.5),))),
                (
                    input = (m_out = NormalMeanVariance(10.0, 1.0), m_ins = ManyOf(NormalMeanVariance()), meta = test_meta),
                    output = FactorizedJoint((NormalMeanVariance(5.0, 0.5),))
                )
            ]
        end
    end

    @testset "id, x ~ MvNormal, y ~ MvNormal" begin
        for enforce in (Val(false), Val(true)), grad in (ForwardDiffGrad(), ForwardDiffGrad(1))
            test_meta = DeltaMeta(method = CVI(StableRNG(123), 1, 1000, Optimisers.Descent(0.007), grad, 20, enforce, true))
            @test_marginalrules [check_type_promotion = false, atol = 1e-2] DeltaFn{identity}(:ins) [
                (
                    input = (m_out = MvGaussianMeanCovariance(ones(2)), m_ins = ManyOf(MvGaussianMeanCovariance(zeros(2))), meta = test_meta),
                    output = FactorizedJoint((MvNormalMeanCovariance(0.5 * ones(2), 0.5 * Matrix(Diagonal(ones(2)))),))
                ),
                (
                    input = (m_out = MvGaussianMeanCovariance(ones(2) * 10), m_ins = ManyOf(MvGaussianMeanCovariance(zeros(2))), meta = test_meta),
                    output = FactorizedJoint((MvNormalMeanCovariance(5.0 * ones(2), 0.5 * Matrix(Diagonal(ones(2)))),))
                ),
                (
                    input = (m_out = MvGaussianMeanCovariance(ones(2), [2 -1; -1 2]), m_ins = ManyOf(MvGaussianMeanCovariance(zeros(2))), meta = test_meta),
                    output = FactorizedJoint(((MvNormalMeanCovariance(0.5 * ones(2), inv([1+2 / 3 1/3; 1/3 1+2 / 3]))),))
                )
            ]
        end
    end
    @testset "f(x) = x + k, x~Normal, y~Normal" begin
        for enforce in (Val(false), Val(true)), grad in (ForwardDiffGrad(), ForwardDiffGrad(1))
            test_meta = DeltaMeta(method = CVI(StableRNG(123), 1, 1000, Optimisers.Descent(0.007), grad, 20, enforce, true))
            @test_marginalrules [check_type_promotion = false, atol = 1e-2] DeltaFn{add_1}(:ins) [
                (input = (m_out = NormalMeanVariance(1, 1), m_ins = ManyOf(NormalMeanVariance()), meta = test_meta), output = FactorizedJoint((NormalMeanVariance(0, 0.5),))),
                (input = (m_out = NormalMeanVariance(2, 1), m_ins = ManyOf(NormalMeanVariance()), meta = test_meta), output = FactorizedJoint((NormalMeanVariance(0.5, 0.5),))),
                (input = (m_out = NormalMeanVariance(10, 1), m_ins = ManyOf(NormalMeanVariance()), meta = test_meta), output = FactorizedJoint((NormalMeanVariance(4.5, 0.5),)))
            ]
        end
    end

    @testset "f(x, y) -> [x, y], x~Normal, y~Normal, out~MvNormal (marginalization)" begin
        for enforce in (Val(false), Val(true)), grad in (ForwardDiffGrad(), ForwardDiffGrad(1))
            test_meta = DeltaMeta(method = CVI(StableRNG(123), 1, 1000, Optimisers.Descent(0.007), grad, 20, enforce, true))
            @test_marginalrules [check_type_promotion = false, atol = 1e-2] DeltaFn{two_into_one}(:ins) [(
                input = (m_out = MvGaussianMeanCovariance(ones(2), [2 0; 0 2]), m_ins = ManyOf(NormalMeanVariance(), NormalMeanVariance(1, 2)), meta = test_meta),
                output = FactorizedJoint((NormalMeanVariance(1 / 3, 2 / 3), NormalMeanVariance(1.0, 1.0)))
            )]
        end
    end

    @testset "f(x) -> x[1], x~MvNormal out~Normal" begin
        for enforce in (Val(false),), grad in (ForwardDiffGrad(), ForwardDiffGrad(1))
            test_meta = DeltaMeta(method = CVI(StableRNG(123), 1, 1000, Optimisers.Descent(0.007), grad, 20, enforce, true))
            @test_marginalrules [check_type_promotion = false, atol = 1e-2] DeltaFn{extract_coordinate}(:ins) [(
                input = (m_out = NormalMeanVariance(0, 1), m_ins = ManyOf(MvGaussianMeanCovariance(ones(2), [1 0; 0 1])), meta = test_meta),
                # output = FactorizedJoint((MvNormalWeightedMeanPrecision(ones(2), [2 0; 0 1]),)),
                output = FactorizedJoint((MvNormalMeanCovariance([0.5, 1.0], [0.5 0.0; 0.0 1.0]),))
            )]
        end
    end

    @testset "id, x~Gamma out~Gamma" begin
        for enforce in (Val(false), Val(true)), grad in (ForwardDiffGrad(), ForwardDiffGrad(1))
            test_meta = DeltaMeta(method = CVI(StableRNG(123), 1, 1000, Optimisers.Descent(0.007), grad, 20, Val(true), true))

            @test_marginalrules [check_type_promotion = false, atol = 1e-1] DeltaFn{identity}(:ins) [
                (input = (m_out = GammaShapeRate(1, 1), m_ins = ManyOf(GammaShapeRate(1, 1)), meta = test_meta), output = FactorizedJoint((Gamma(1, 1 / 2),))),
                (input = (m_out = GammaShapeRate(1, 1), m_ins = ManyOf(GammaShapeRate(1, 2)), meta = test_meta), output = FactorizedJoint((Gamma(1, 1 / 3),)))
            ]
        end

        for enforce in (Val(false), Val(true)), grad in (ForwardDiffGrad(), ForwardDiffGrad(1))
            test_meta = DeltaMeta(method = CVI(StableRNG(123), 1, 2000, Optimisers.Descent(0.003), grad, 20, Val(true), true))
            @test_marginalrules [check_type_promotion = false, atol = 1e-1] DeltaFn{identity}(:ins) [
                (input = (m_out = GammaShapeRate(2, 1), m_ins = ManyOf(GammaShapeRate(1, 2)), meta = test_meta), output = FactorizedJoint((Gamma(2, 1 / 3),))),
                (input = (m_out = GammaShapeRate(2, 2), m_ins = ManyOf(GammaShapeRate(2, 3)), meta = test_meta), output = FactorizedJoint((Gamma(3, 1 / 5),)))
            ]
        end
    end
end
