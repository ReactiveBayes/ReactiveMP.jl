
@testitem "rules:SoftDot:x" begin
    using ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions

    import ReactiveMP: @test_rules

    # Semi-exhaustive combination of input types
    #=
    0: PointMass
    1: UnivariateNormalDistributionsFamily (NormalMeanVariance)
    2: MultivariateNormalDistributionsFamily (MvNormalMeanCovariance)
    3: GammaDistributionsFamily (GammaShapeScale)
    y ∈ {0, 1}
    x ∈ {0, 1, 2}
    θ ∈ {0, 1, 2}
    γ ∈ {0, 3}
    =#
    #=
    #Copy & paste the functionality below to locally test
    f(q_y, q_θ, q_γ) = begin
       m_y, V_y = mean_cov(q_y)
       m_x, V_x = mean_cov(q_θ)
       m_γ = mean(q_γ)
       D = m_γ*(V_x + m_x*m_x')
       z = m_γ*m_x*m_y
       return z, D
    end
    f(PointMass(3.0), NormalMeanVariance(2.0, 7.0), PointMass(5.0)) = (30.0, 55.0) = (z, D)
        ⇒ NormalWeightedMeanPrecision()
    =#
    @testset "VMP: Mean-field" begin
        # 000
        @testset "(q_y::PointMass, q_θ::PointMass, q_γ::PointMass)" begin
            @test_rules [check_type_promotion = true] SoftDot(:x, Marginalisation) [(
                input = (q_y = PointMass(3.0), q_θ = PointMass(5.0), q_γ = PointMass(2.0)), output = NormalWeightedMeanPrecision(30.0, 50.0)
            )]
        end

        # 003
        @testset "(q_y::PointMass, q_θ::PointMass, q_γ::GammaShapeScale)" begin
            @test_rules [check_type_promotion = true] SoftDot(:x, Marginalisation) [(
                input = (q_y = PointMass(3.0), q_θ = PointMass(5.0), q_γ = GammaShapeScale(2.0, 7.0)), output = NormalWeightedMeanPrecision(210.0, 350.0)
            )]
        end

        # 010
        @testset "(q_y::PointMass, q_θ::NormalMeanVariance, q_γ::PointMass)" begin
            @test_rules [check_type_promotion = true] SoftDot(:x, Marginalisation) [(
                input = (q_y = PointMass(3.0), q_θ = NormalMeanVariance(2.0, 7.0), q_γ = PointMass(2.0)), output = NormalWeightedMeanPrecision(12.0, 22.0)
            )]
        end

        # 013
        @testset "(q_y::PointMass, q_θ::NormalMeanVariance, q_γ::GammaShapeScale)" begin
            @test_rules [check_type_promotion = true] SoftDot(:x, Marginalisation) [(
                input = (q_y = PointMass(3.0), q_θ = NormalMeanVariance(2.0, 7.0), q_γ = GammaShapeScale(5.0, 11.0)), output = NormalWeightedMeanPrecision(330.0, 605.0)
            )]
        end

        # 020
        @testset "(q_y::PointMass, q_θ::MvNormalMeanCovariance, q_γ::PointMass)" begin
            @test_rules [check_type_promotion = true] SoftDot(:x, Marginalisation) [(
                input = (q_y = PointMass(3.0), q_θ = MvNormalMeanCovariance([5.0, 9.0], [11.0 13.0; 17.0 19.0]), q_γ = PointMass(2.0)),
                output = MvNormalWeightedMeanPrecision([30.0, 54.0], [72.0 116.0; 124.0 200.0])
            )]
        end

        # 023
        @testset "(q_y::PointMass, q_θ::MvNormalMeanCovariance, q_γ::GammaShapeScale)" begin
            @test_rules [check_type_promotion = true] SoftDot(:x, Marginalisation) [(
                input = (q_y = PointMass(3.0), q_θ = MvNormalMeanCovariance([5.0, 9.0], [11.0 13.0; 17.0 19.0]), q_γ = GammaShapeScale(7.0, 23.0)),
                output = MvNormalWeightedMeanPrecision([2415.0, 4347.0], [5796.0 9338.0; 9982.0 16100.0])
            )]
        end

        # 100
        @testset "(q_y::NormalMeanVariance, q_θ::PointMass, q_γ::PointMass)" begin
            @test_rules [check_type_promotion = true] SoftDot(:x, Marginalisation) [(
                input = (q_y = NormalMeanVariance(3.0, 7.0), q_θ = PointMass(5.0), q_γ = PointMass(2.0)), output = NormalWeightedMeanPrecision(30.0, 50.0)
            )]
        end

        # 103
        @testset "(q_y::NormalMeanVariance, q_θ::PointMass, q_γ::GammaShapeScale)" begin
            @test_rules [check_type_promotion = true] SoftDot(:x, Marginalisation) [(
                input = (q_y = NormalMeanVariance(3.0, 7.0), q_θ = PointMass(5.0), q_γ = GammaShapeScale(2.0, 7.0)), output = NormalWeightedMeanPrecision(210.0, 350.0)
            )]
        end

        # 110
        @testset "(q_y::NormalMeanVariance, q_θ::NormalMeanVariance, q_γ::PointMass)" begin
            @test_rules [check_type_promotion = true] SoftDot(:x, Marginalisation) [(
                input = (q_y = NormalMeanVariance(3.0, 7.0), q_θ = NormalMeanVariance(2.0, 7.0), q_γ = PointMass(2.0)), output = NormalWeightedMeanPrecision(12.0, 22.0)
            )]
        end

        # 113
        @testset "(q_y::NormalMeanVariance, q_θ::NormalMeanVariance, q_γ::GammaShapeScale)" begin
            @test_rules [check_type_promotion = true] SoftDot(:x, Marginalisation) [(
                input = (q_y = NormalMeanVariance(3.0, 7.0), q_θ = NormalMeanVariance(2.0, 7.0), q_γ = GammaShapeScale(5.0, 11.0)),
                output = NormalWeightedMeanPrecision(330.0, 605.0)
            )]
        end

        # 120
        @testset "(q_y::NormalMeanVariance, q_θ::MvNormalMeanCovariance, q_γ::PointMass)" begin
            @test_rules [check_type_promotion = true] SoftDot(:x, Marginalisation) [(
                input = (q_y = NormalMeanVariance(3.0, 7.0), q_θ = MvNormalMeanCovariance([5.0, 9.0], [11.0 13.0; 17.0 19.0]), q_γ = PointMass(2.0)),
                output = MvNormalWeightedMeanPrecision([30.0, 54.0], [72.0 116.0; 124.0 200.0])
            )]
        end

        # 123
        @testset "(q_y::NormalMeanVariance, q_θ::MvNormalMeanCovariance, q_γ::GammaShapeScale)" begin
            @test_rules [check_type_promotion = true] SoftDot(:x, Marginalisation) [(
                input = (q_y = NormalMeanVariance(3.0, 7.0), q_θ = MvNormalMeanCovariance([5.0, 9.0], [11.0 13.0; 17.0 19.0]), q_γ = GammaShapeScale(7.0, 23.0)),
                output = MvNormalWeightedMeanPrecision([2415.0, 4347.0], [5796.0 9338.0; 9982.0 16100.0])
            )]
        end
    end

    # TODO: these errors have to be caught in the implementations themselves. The error type and message itself will not provide any information or might not match.
    @testset "VMP: Incorrect Inputs" begin
        # 2**: INCORRECT (y cannot be Mv)
        @testset "(q_y::MvNormalMeanCovariance, q_θ::CORRECT, q_γ::CORRECT)" begin
            @test_throws MethodError @call_rule SoftDot(:x, Marginalisation) (
                q_y = MvNormalMeanCovariance([3.0, 7.0], [11.0, 13.0]), q_θ = NormalMeanVariance(7.0, 11.0), q_γ = GammaShapeScale(13.0, 5.0)
            )
        end
        # NOTE: γ can theoretically be Any, so also NormalMeanVariance
    end

    @testset "VMP: structured rules" begin
        @testset "(m_y::NormalMeanVariance, q_θ::NormalMeanVariance, q_γ::Any)" begin
            @test_rules [check_type_promotion = true] SoftDot(:x, Marginalisation) [
                (input = (m_y = NormalMeanVariance(1.0, 1.0), q_θ = NormalMeanVariance(1.0, 1.0), q_γ = GammaShapeRate(1.0, 1.0)), output = NormalWeightedMeanPrecision(0.5, 1.5)),
                (
                    input = (m_y = NormalWeightedMeanPrecision(1.0, 1.0), q_θ = NormalMeanPrecision(1.0, 2.0), q_γ = GammaShapeScale(1.0, 1.0)),
                    output = NormalWeightedMeanPrecision(0.5, 1.0)
                )
            ]
        end

        @testset "(m_y::UnivariateNormalDistributionsFamily, q_θ::MultivariateNormalDistributionsFamily, q_γ::Any)" begin
            order = 2
            @test_rules [check_type_promotion = false] SoftDot(:x, Marginalisation) [
                (
                    input = (m_y = NormalMeanVariance(0.0, 1.0), q_θ = MvNormalMeanCovariance(ones(order), diageye(order)), q_γ = GammaShapeRate(1.0, 1.0)),
                    output = MvNormalWeightedMeanPrecision([0.0, 0.0], [1.5 0.5; 0.5 1.5])
                ),
                (
                    input = (m_y = NormalMeanVariance(1.0, 1.0), q_θ = MvNormalMeanCovariance(zeros(order), diageye(order)), q_γ = GammaShapeScale(1.0, 1.0)),
                    output = MvNormalWeightedMeanPrecision([0.0, 0.0], [1.0 0.0; 0.0 1.0])
                ),
                (
                    input = (m_y = NormalMeanVariance(1.0, 1.0), q_θ = MvNormalMeanCovariance(ones(order), diageye(order)), q_γ = Gamma(1.0, 1.0)),
                    output = MvNormalWeightedMeanPrecision([0.5, 0.5], [1.5 0.5; 0.5 1.5])
                )
            ]
        end
    end
end # testset
