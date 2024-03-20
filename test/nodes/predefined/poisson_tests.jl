
@testitem "PoissonNode" begin
    using ReactiveMP, Random, BayesBase, ExponentialFamily

    @testset "Average energy" begin

        for l in 1:20, k in 1:20
            @test isapprox(
                score(AverageEnergy(), Poisson, Val{(:out, :l)}(), (Marginal(PointMass(k), false, false, nothing), Marginal(PointMass(l), false, false, nothing)), nothing),
                -logpdf(Poisson(l), k),
                rtol = 1e-12
            )
        end

        for k in 1:100
            @test isapprox(
                score(AverageEnergy(), Poisson, Val{(:out, :l)}(), (Marginal(Poisson(k), false, false, nothing), Marginal(PointMass(k), false, false, nothing)), nothing),
                entropy(Poisson(k)),
                rtol = 1e-3
            )
        end

        for k in 101:110
            @test isapprox(
                score(AverageEnergy(), Poisson, Val{(:out, :l)}(), (Marginal(Poisson(k), false, false, nothing), Marginal(PointMass(k), false, false, nothing)), nothing),
                entropy(Poisson(k)),
                rtol = 1e-1
            )
        end
    end
end
