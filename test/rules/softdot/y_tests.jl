
@testitem "rules:SoftDot:y" begin
    using ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions

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
    #=
    #Copy & paste the functionality below to locally test
    f(q_θ, q_x, q_γ) = (mean(q_θ)'mean(q_x), mean(q_γ))
    f(PointMass(3.0), PointMass(11.0), GammaShapeRate(7.0, 5.0)) = (33.0, 1.4)
    =#
    @testset "VMP: Mean-field" begin
        # 000
        @testset "(q_θ::PointMass, q_x::PointMass, q_γ::PointMass" begin
            @test_rules [check_type_promotion = true] SoftDot(:y, Marginalisation) [(
                input = (q_θ = PointMass(3.0), q_x = PointMass(11.0), q_γ = PointMass(7.0)), output = NormalMeanPrecision(33.0, 7.0)
            )]
        end

        # 003
        @testset "(q_θ::PointMass, q_x::PointMass, q_γ::{<:GammaDistributionsFamily)" begin
            @test_rules [check_type_promotion = true] SoftDot(:y, Marginalisation) [
                (input = (q_θ = PointMass(3.0), q_x = PointMass(11.0), q_γ = GammaShapeRate(7.0, 5.0)), output = NormalMeanPrecision(33.0, 1.4)),
                (input = (q_θ = PointMass(3.0), q_x = PointMass(11.0), q_γ = GammaShapeScale(7.0, 5.0)), output = NormalMeanPrecision(33.0, 35.0))
            ]
        end

        # 110
        @testset "(q_θ::NormalMeanVariance, q_x::NormalMeanVariance, q_γ::{<:GammaDistributionsFamily})" begin
            @test_rules [check_type_promotion = true] SoftDot(:y, Marginalisation) [(
                input = (q_θ = NormalMeanVariance(3.0, 17.0), q_x = NormalMeanVariance(7.0, 11.0), q_γ = PointMass(13.0)), output = NormalMeanPrecision(21.0, 13.0)
            )]
        end

        # 113
        @testset "(q_θ::NormalMeanVariance, q_x::NormalMeanVariance, q_γ::{<:GammaDistributionsFamily})" begin
            @test_rules [check_type_promotion = true] SoftDot(:y, Marginalisation) [
                (input = (q_θ = NormalMeanVariance(3.0, 17.0), q_x = NormalMeanVariance(7.0, 11.0), q_γ = GammaShapeRate(13.0, 5.0)), output = NormalMeanPrecision(21.0, 2.6)),
                (input = (q_θ = NormalMeanVariance(3.0, 17.0), q_x = NormalMeanVariance(7.0, 11.0), q_γ = GammaShapeScale(13.0, 5.0)), output = NormalMeanPrecision(21.0, 65.0))
            ]
        end

        # 220
        @testset "(q_θ::MvNormalMeanCovariance, q_x::MvNormalMeanCovariance, q_γ::{<:GammaDistributionsFamily})" begin
            @test_rules [check_type_promotion = true] SoftDot(:y, Marginalisation) [(
                input = (q_θ = MvNormalMeanCovariance([3.0, 7.0], [11.0, 13.0]), q_x = MvNormalMeanCovariance([17.0, 19.0], [23.0, 29.0]), q_γ = PointMass(31.0)),
                output = NormalMeanPrecision(184.0, 31.0)
            )]
        end

        # 223
        @testset "(q_θ::MvNormalMeanCovariance, q_x::MvNormalMeanCovariance, q_γ::{<:GammaDistributionsFamily})" begin
            @test_rules [check_type_promotion = true] SoftDot(:y, Marginalisation) [
                (
                    input = (q_θ = MvNormalMeanCovariance([3.0, 7.0], [11.0, 13.0]), q_x = MvNormalMeanCovariance([17.0, 19.0], [23.0, 29.0]), q_γ = GammaShapeRate(31.0, 5.0)),
                    output = NormalMeanPrecision(184.0, 6.2)
                ),
                (
                    input = (q_θ = MvNormalMeanCovariance([3.0, 7.0], [11.0, 13.0]), q_x = MvNormalMeanCovariance([17.0, 19.0], [23.0, 29.0]), q_γ = GammaShapeScale(31.0, 5.0)),
                    output = NormalMeanPrecision(184.0, 155.0)
                )
            ]
        end
    end # testset: mean-field

    # TODO: these errors have to be caught in the implementations themselves. The error type and message itself will not provide any information or might not match.
    @testset "VMP: Incorrect Inputs" begin
        # 02*: INCORRECT (θ and x have to be of the same dimension)
        @testset "(q_θ::PointMass, q_x::MvNormalMeanCovariance, q_γ::CORRECT)" begin
            @test_throws MethodError @call_rule SoftDot(:y, Marginalisation) (
                q_θ = PointMass(7.0), q_x = MvNormalMeanCovariance([3.0, 7.0], [11.0, 13.0]), q_γ = GammaShapeRate(13.0, 5.0)
            )
        end

        # 12*: INCORRECT (θ and x have to be of the same dimension)
        @testset "(q_θ::NormalMeanVariance, q_x::MvNormalMeanCovariance, q_γ::CORRECT)" begin
            @test_throws MethodError @call_rule SoftDot(:y, Marginalisation) (
                q_θ = NormalMeanVariance(7.0, 11.0), q_x = MvNormalMeanCovariance([3.0, 7.0], [11.0, 13.0]), q_γ = GammaShapeRate(13.0, 5.0)
            )
        end

        # 21*: INCORRECT (θ and x have to be of the same dimension)
        @testset "(q_θ::MvNormalMeanCovariance, q_x::NormalMeanVariance, q_γ::CORRECT)" begin
            @test_throws MethodError @call_rule SoftDot(:y, Marginalisation) (
                q_θ = MvNormalMeanCovariance([3.0, 7.0], [11.0, 13.0]), q_x = NormalMeanVariance(7.0, 11.0), q_γ = GammaShapeRate(13.0, 5.0)
            )
        end
    end

    @testset "VMP: structured rules" begin
        @testset "(q_θ::NormalMeanVariance, m_x::NormalMeanVariance, q_γ::Any" begin
            @test_rules [check_type_promotion = true] SoftDot(:y, Marginalisation) [
                (input = (q_θ = PointMass(3.0), q_x = PointMass(11.0), q_γ = GammaShapeRate(7.0, 5.0)), output = NormalMeanPrecision(33.0, 1.4)),
                (input = (q_θ = PointMass(3.0), q_x = PointMass(11.0), q_γ = GammaShapeScale(7.0, 5.0)), output = NormalMeanPrecision(33.0, 35.0))
            ]

            @test_rules [check_type_promotion = true] SoftDot(:y, Marginalisation) [
                (input = (m_x = NormalMeanVariance(1.0, 1.0), q_θ = NormalMeanVariance(1.0, 1.0), q_γ = GammaShapeRate(1.0, 1.0)), output = NormalMeanVariance(0.5, 1.5)),
                (
                    input = (m_x = NormalWeightedMeanPrecision(1.0, 1.0), q_θ = NormalMeanPrecision(1.0, 2.0), q_γ = GammaShapeScale(2.0, 1.0)),
                    output = NormalMeanVariance(0.5, 1.0)
                )
            ]
        end

        @testset "(q_θ::MvNormalMeanCovariance, m_x::MvNormalMeanCovariance, q_γ::Any" begin
            order = 2
            @test_rules [check_type_promotion = true] SoftDot(:y, Marginalisation) [
                (
                    input = (
                        m_x = MvNormalMeanCovariance(ones(order), diageye(order)), q_θ = MvNormalMeanCovariance(zeros(order), diageye(order)), q_γ = GammaShapeScale(1.0, 1.0)
                    ),
                    output = NormalMeanVariance(0.0, 1.0)
                ),
                (
                    input = (m_x = MvNormalMeanCovariance(ones(order), diageye(order)), q_θ = MvNormalMeanCovariance(ones(order), diageye(order)), q_γ = Gamma(1.0, 1.0)),
                    output = NormalMeanVariance(1.0, 2.0)
                )
            ]
        end
    end
end # testset
