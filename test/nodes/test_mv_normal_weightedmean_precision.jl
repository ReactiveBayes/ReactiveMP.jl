module MvNormalWeightedMeanPrecisionNodeTest

using Test, ReactiveMP, Random, BayesBase, ExponentialFamily

import ReactiveMP: make_node

@testset "MvNormalWeightedMeanPrecisionNodeTest" begin
    @testset "Creation" begin
        node = make_node(MvNormalWeightedMeanPrecision)

        @test functionalform(node) === MvNormalWeightedMeanPrecision
        @test sdtype(node) === Stochastic()
        @test name.(interfaces(node)) === (:out, :ξ, :Λ)
        @test factorisation(node) === ((1, 2, 3),)
        @test localmarginalnames(node) === (:out_ξ_Λ,)
        @test metadata(node) === nothing

        node = make_node(MvNormalWeightedMeanPrecision, FactorNodeCreationOptions(nothing, 1, nothing))

        @test functionalform(node) === MvNormalWeightedMeanPrecision
        @test sdtype(node) === Stochastic()
        @test name.(interfaces(node)) === (:out, :ξ, :Λ)
        @test factorisation(node) === ((1, 2, 3),)
        @test localmarginalnames(node) === (:out_ξ_Λ,)
        @test metadata(node) === 1

        node = make_node(MvNormalWeightedMeanPrecision, FactorNodeCreationOptions(((1,), (2, 3)), nothing, nothing))

        @test functionalform(node) === MvNormalWeightedMeanPrecision
        @test sdtype(node) === Stochastic()
        @test name.(interfaces(node)) === (:out, :ξ, :Λ)
        @test factorisation(node) === ((1,), (2, 3))
        @test localmarginalnames(node) === (:out, :ξ_Λ)
        @test metadata(node) === nothing
    end

    @testset "AverageEnergy" begin
        begin
            q_out = PointMass([1.0, 1.0])
            q_Λ   = PointMass([1/2 0.0; 0.0 1/2])
            q_ξ   = PointMass(mean(q_Λ) * [2.0, 2.0])

            for N in (MvNormalMeanPrecision, MvNormalMeanCovariance, MvNormalWeightedMeanPrecision)
                marginals = (Marginal(q_out, false, false, nothing), Marginal(q_ξ, false, false, nothing), Marginal(q_Λ, false, false, nothing))
                @test score(AverageEnergy(), MvNormalWeightedMeanPrecision, Val{(:out, :ξ, :Λ)}(), marginals, nothing) ≈ 3.0310242469692907
            end
        end

        # begin

        #     # q_out = MvNormalMeanCovariance([0.34777017741128646, 0.478300212208703], [0.36248838736696753 0.42718825177834396; 0.42718825177834396 0.6613345138178703])
        #     # q_Λ   = PointMass([0.9575163948991356 0.952247049670764; 0.952247049670764 1.001998675194526])
        #     # q_ξ   = PointMass(mean(q_Λ)*[0.28114649146383175, 0.7773140083754762])

        #     q_out = PointMass([1.0, 1.0])
        #     q_Λ   = Wishart(4, [1/2 0.0; 0.0 1/2])
        #     # q_ξ   = PointMass(mean(q_Λ)*[2.0, 2.0])
        #     q_μ   = PointMass([2.0, 2.0])

        #     for N in (MvNormalMeanPrecision, MvNormalMeanCovariance, MvNormalWeightedMeanPrecision), G in (Wishart,)
        #         marginals1 = (Marginal(q_out, false, false, nothing), Marginal(q_ξ, false, false, nothing), Marginal(convert(G, q_Λ), false, false, nothing))
        #         marginals2 = (Marginal(q_out, false, false, nothing), Marginal(q_μ, false, false, nothing), Marginal(convert(G, q_Λ), false, false, nothing))
        #         @test score(AverageEnergy(), MvNormalWeightedMeanPrecision, Val{(:out, :ξ, :Λ)}(), marginals1, nothing) ≈ score(AverageEnergy(), MvNormalMeanPrecision, Val{(:out, :μ, :Λ)}(), marginals2, nothing)
        #     end
        # end

        # begin
        #     μ_out = [0.8625589157256603, 0.6694783342639599]
        #     Λ_out = [1.0014322413749484 0.7989099036521625; 0.7989099036521625 1.0976639268696966]
        #     μ_ξ   = [0.9334416739853251, 0.38318522701093105]
        #     Λ_ξ   = [0.21867945696266933 0.5704895781120056; 0.5704895781120056 1.5321190185800933]
        #     q_out = MvNormalWeightedMeanPrecision(Λ_out * μ_out, Λ_out)
        #     q_ξ   = MvNormalWeightedMeanPrecision(Λ_ξ * μ_ξ, Λ_ξ)
        #     q_Λ   = Wishart(4, [1.2509510680086597 1.3662166757006484; 1.3662166757006484 1.4925451500802907])

        #     for N1 in (MvNormalMeanPrecision, MvNormalMeanCovariance, MvNormalWeightedMeanPrecision),
        #         N2 in (MvNormalMeanPrecision, MvNormalMeanCovariance, MvNormalWeightedMeanPrecision),
        #         G in (Wishart,)

        #         marginals = (
        #             Marginal(convert(N1, q_out), false, false, nothing), Marginal(convert(N2, q_ξ), false, false, nothing), Marginal(convert(G, q_Λ), false, false, nothing)
        #         )
        #         @test score(AverageEnergy(), MvNormalWeightedMeanPrecision, Val{(:out, :ξ, :Λ)}(), marginals, nothing) ≈ 114.57678973411974
        #     end
        # end

    end
end
end
