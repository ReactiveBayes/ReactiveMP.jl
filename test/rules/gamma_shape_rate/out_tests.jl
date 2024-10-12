
@testitem "rules:GammaShapeRate:out" begin
    using ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions

    import ReactiveMP: @test_rules

    @testset "Belief Propagation: (m_α::Any, m_θ::Any)" begin
        @test_rules [check_type_promotion = true] GammaShapeRate(:out, Marginalisation) [
            (input = (m_α = PointMass(1.0), m_β = PointMass(2.0)), output = GammaShapeRate(1.0, 2.0)),
            (input = (m_α = PointMass(3.0), m_β = PointMass(3.0)), output = GammaShapeRate(3.0, 3.0)),
            (input = (m_α = PointMass(42.0), m_β = PointMass(42.0)), output = GammaShapeRate(42.0, 42.0))
        ]
    end

    @testset "Variational Message Passing: (q_α::Any, q_β::Any)" begin
        @test_rules [check_type_promotion = true] GammaShapeRate(:out, Marginalisation) [
            (input = (q_α = PointMass(1.0), q_β = PointMass(2.0)), output = GammaShapeRate(1.0, 2.0)),
            (input = (q_α = GammaShapeScale(1.0, 1.0), q_β = GammaShapeRate(1.0, 1.0)), output = GammaShapeRate(1.0, 1.0))
        ]
    end
end # testset
