
@testitem "PoissonNode" begin
    using ReactiveMP, Random, BayesBase, ExponentialFamily

    import ReactiveMP: make_node

    @testset "Creation" begin
        node = make_node(Poisson)

        @test functionalform(node) === Poisson
        @test sdtype(node) === Stochastic()
        @test name.(interfaces(node)) === (:out, :l)
        @test factorisation(node) === ((1, 2),)
        @test localmarginalnames(node) === (:out_l,)
        @test metadata(node) === nothing

        node = make_node(Poisson, FactorNodeCreationOptions(nothing, 1, nothing))

        @test metadata(node) === 1
    end

    @testset "Average energy" begin
        node = ReactiveMP.make_node(Poisson)

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
