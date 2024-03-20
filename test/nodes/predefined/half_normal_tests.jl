
@testitem "HalfNormalNode" begin
    using ReactiveMP, Random, BayesBase, ExponentialFamily

    import ReactiveMP: make_node

    @testset "AverageEnergy" begin
        begin
            q_out = GammaShapeRate(2.0, 1.0)
            q_v   = PointMass(2.0)

            marginals = (Marginal(q_out, false, false, nothing), Marginal(q_v, false, false, nothing))

            @test score(AverageEnergy(), HalfNormal, Val{(:out, :v)}(), marginals, nothing) ≈ 2.072364942925
        end
        begin
            q_out = GammaShapeScale(2.0, 1.0)
            q_v   = PointMass(2.0)

            marginals = (Marginal(q_out, false, false, nothing), Marginal(q_v, false, false, nothing))

            @test score(AverageEnergy(), HalfNormal, Val{(:out, :v)}(), marginals, nothing) ≈ 2.072364942925
        end

        begin
            q_out = GammaInverse(3.0, 1.0)
            q_v   = PointMass(2.0)

            marginals = (Marginal(q_out, false, false, nothing), Marginal(q_v, false, false, nothing))

            @test score(AverageEnergy(), HalfNormal, Val{(:out, :v)}(), marginals, nothing) ≈ 0.6973649429247
        end
    end # testset: AverageEnergy
end # testset
