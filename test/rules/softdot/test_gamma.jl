module RulesSoftDotGammaTest

using Test, ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions

import ReactiveMP: @test_rules

# Semi-exhaustive combination of input types
#=
0: PointMass
1: UnivariateNormalDistributionsFamily (NormalMeanVariance)
2: MultivariateNormalDistributionsFamily (MvNormalMeanCovariance)
3: GammaDistributionsFamily
y ∈ {0, 1}
x ∈ {0, 1, 2}
θ ∈ {0, 1, 2}
γ ∈ {0, 3}
=#
@testset "rules:SoftDot:γ" begin
    #=
    #Copy & paste the functionality below to locally test (f returns B in Γ(3/2, B/2))
    f(q_y,q_θ,q_x) = begin
        m_y, V_y = mean_cov(q_y)
        m_θ, V_θ = mean_cov(q_θ)
        m_x, V_x = mean_cov(q_x)
        return 0.5*(V_y+m_y^2 - 2*m_y*m_θ'm_x + ReactiveMP.mul_trace(V_x, V_θ) + m_θ'(V_x+m_x*m_x')*m_θ + m_x'V_θ*m_x)
    end
    f(PointMass(3.0), NormalMeanVariance(2.0, 7.0), PointMass(5.0))
    =#
    @testset "VMP: Mean-field" begin
        # 000
        @testset "(q_y::PointMass, q_θ::PointMass, q_x::PointMass)" begin
            @test_rules [check_type_promotion = true] SoftDot(:γ, Marginalisation) [(
                input = (q_y = PointMass(3.0), q_θ = PointMass(5.0), q_x = PointMass(2.0)), output = GammaShapeRate(3 / 2, 49 / 2)
            )]
        end

        # 001
        @testset "(q_y::PointMass, q_θ::PointMass, q_x::NormalMeanVariance)" begin
            @test_rules [check_type_promotion = true] SoftDot(:γ, Marginalisation) [(
                input = (q_y = PointMass(3.0), q_θ = PointMass(5.0), q_x = NormalMeanVariance(2.0, 7.0)), output = GammaShapeRate(3 / 2, 224 / 2)
            )]
        end

        # 010
        @testset "(q_y::PointMass, q_θ::NormalMeanVariance, q_x::PointMass)" begin
            @test_rules [check_type_promotion = true] SoftDot(:γ, Marginalisation) [(
                input = (q_y = PointMass(3.0), q_θ = NormalMeanVariance(2.0, 7.0), q_x = PointMass(5.0)), output = GammaShapeRate(3 / 2, 224 / 2)
            )]
        end

        # 011
        @testset "(q_y::PointMass, q_θ::NormalMeanVariance, q_x::NormalMeanVariance)" begin
            @test_rules [check_type_promotion = true] SoftDot(:γ, Marginalisation) [(
                input = (q_y = PointMass(3.0), q_θ = NormalMeanVariance(2.0, 7.0), q_x = NormalMeanVariance(5.0, 11.0)), output = GammaShapeRate(3 / 2, 345 / 2)
            )]
        end

        # 100
        @testset "(q_y::NormalMeanVariance, q_θ::PointMass, q_x::PointMass)" begin
            @test_rules [check_type_promotion = true] SoftDot(:γ, Marginalisation) [(
                input = (q_y = NormalMeanVariance(2.0, 3.0), q_θ = PointMass(5.0), q_x = PointMass(7.0)), output = GammaShapeRate(3 / 2, 1092 / 2)
            )]
        end

        # 101
        @testset "(q_y::NormalMeanVariance, q_θ::PointMass, q_x::NormalMeanVariance)" begin
            @test_rules [check_type_promotion = true] SoftDot(:γ, Marginalisation) [(
                input = (q_y = NormalMeanVariance(3.0, 7.0), q_θ = PointMass(2.0), q_x = NormalMeanVariance(5.0, 9.0)), output = GammaShapeRate(3 / 2, 92 / 2)
            )]
        end

        # 110
        @testset "(q_y::NormalMeanVariance, q_θ::NormalMeanVariance, q_x::PointMass)" begin
            @test_rules [check_type_promotion = true] SoftDot(:γ, Marginalisation) [(
                input = (q_y = NormalMeanVariance(3.0, 7.0), q_θ = NormalMeanVariance(5.0, 9.0), q_x = PointMass(2.0)), output = GammaShapeRate(3 / 2, 92 / 2)
            )]
        end

        # 111
        @testset "(q_y::NormalMeanVariance, q_θ::NormalMeanVariance, q_x::NormalMeanVariance)" begin
            @test_rules [check_type_promotion = true] SoftDot(:γ, Marginalisation) [(
                input = (q_y = NormalMeanVariance(3.0, 7.0), q_θ = NormalMeanVariance(5.0, 9.0), q_x = NormalMeanVariance(11.0, 13.0)), output = GammaShapeRate(3 / 2, 4242 / 2)
            )]
        end

        # 122
        @testset "(q_y::NormalMeanVariance, q_θ::MvNormalMeanCovariance, q_x::MvNormalMeanCovariance)" begin
            @test_rules [check_type_promotion = true] SoftDot(:γ, Marginalisation) [(
                input = (
                    q_y = NormalMeanVariance(3.0, 7.0),
                    q_θ = MvNormalMeanCovariance([5.0, 9.0], [11.0 13.0; 17.0 19.0]),
                    q_x = MvNormalMeanCovariance([23.0, 29.0], [31.0 37.0; 41.0 43.0])
                ),
                output = GammaShapeRate(3 / 2, 191032 / 2)
            )]
        end
    end # testset: mean-field

    # TODO: these errors have to be caught in the implementations themselves. The error type and message itself will not provide any information or might not match.
    @testset "VMP: Incorrect Inputs" begin
        # 2**: INCORRECT (y cannot be Mv)
        @testset "(q_y::MvNormalMeanCovariance, q_θ::CORRECT, q_x::CORRECT)" begin
            @test_throws DimensionMismatch @call_rule SoftDot(:γ, Marginalisation) (
                q_y = MvNormalMeanCovariance([3.0, 7.0], [11.0, 13.0]), q_θ = NormalMeanVariance(7.0, 11.0), q_x = NormalMeanVariance(13.0, 5.0)
            )
        end
        # *02: INCORRECT (θ and x have to have the same dimensions)
        @testset "(q_y::CORRECT, q_θ::PointMass, q_x::MvNormalMeanCovariance)" begin
            @test_throws MethodError @call_rule SoftDot(:γ, Marginalisation) (
                q_y = NormalMeanVariance(3.0, 7.0), q_θ = PointMass(7.0), q_x = MvNormalMeanCovariance([3.0, 7.0], [11.0, 13.0])
            )
        end

        # *20: INCORRECT (θ and x have to have the same dimensions)
        @testset "(q_y::CORRECT, q_θ::MvNormalMeanCovariance, q_x::PointMass)" begin
            @test_throws MethodError @call_rule SoftDot(:γ, Marginalisation) (
                q_y = NormalMeanVariance(3.0, 7.0), q_θ = MvNormalMeanCovariance([3.0, 7.0], [11.0, 13.0]), q_x = PointMass(7.0)
            )
        end

        # *12: INCORRECT (θ and x have to have the same dimensions)
        @testset "(q_y::CORRECT, q_θ::NormalMeanVariance, q_x::MvNormalMeanCovariance)" begin
            @test_throws MethodError @call_rule SoftDot(:γ, Marginalisation) (
                q_y = NormalMeanVariance(3.0, 7.0), q_θ = NormalMeanVariance(7.0, 11.0), q_x = MvNormalMeanCovariance([3.0, 7.0], [11.0, 13.0])
            )
        end
        # *21: INCORRECT (θ and x have to have the same dimensions)
        @testset "(q_y::CORRECT, q_θ::MvNormalMeanCovariance, q_x::NormalMeanVariance)" begin
            @test_throws MethodError @call_rule SoftDot(:γ, Marginalisation) (
                q_y = NormalMeanVariance(3.0, 7.0), q_θ = MvNormalMeanCovariance([3.0, 7.0], [11.0, 13.0]), q_x = NormalMeanVariance(7.0, 11.0)
            )
        end
    end # testset: mean-field
end # testset
end # module
