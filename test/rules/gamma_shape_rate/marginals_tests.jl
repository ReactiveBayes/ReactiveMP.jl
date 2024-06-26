@testitem "marginalrules:GammaShapeRate" begin
    using ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions

    import ReactiveMP: @test_marginalrules

    @testset "out_α_β: (m_out::GammaDistributionsFamily, m_α::PointMass, m_β::PointMass)" begin
        @test_marginalrules [check_type_promotion = false] GammaShapeRate(:out_α_β) [
            (
                input = (m_out = GammaShapeRate(1.0, 2.0), m_α = PointMass(1.0), m_β = PointMass(2.0)),
                output = (out = GammaShapeRate(1.0, 4.0), α = PointMass(1.0), β = PointMass(2.0))
            ),
            (
                input = (m_out = GammaShapeScale(2.0, 2.0), m_α = PointMass(2.0), m_β = PointMass(3.0)),
                output = (out = GammaShapeRate(3.0, 3.5), α = PointMass(2.0), β = PointMass(3.0))
            ),
            (
                input = (m_out = GammaShapeRate(2.0, 3.0), m_α = PointMass(1.0), m_β = PointMass(3.0)),
                output = (out = GammaShapeRate(2.0, 6.0), α = PointMass(1.0), β = PointMass(3.0))
            )
        ]
    end
end
