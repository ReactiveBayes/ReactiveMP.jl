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
            q_ξ   = MvNormalWeightedMeanPrecision([1.0, 1.0], [1.0 0.0; 0.0 1.0])
            q_Λ   = Wishart(3, [2.0 0.0; 0.0 2.0])

            for N in (MvNormalMeanPrecision, MvNormalMeanCovariance, MvNormalWeightedMeanPrecision), G in (Wishart,)
                marginals = (Marginal(q_out, false, false, nothing), Marginal(convert(N, q_ξ), false, false, nothing), Marginal(convert(G, q_Λ), false, false, nothing))
                @test score(AverageEnergy(), MvNormalWeightedMeanPrecision, Val{(:out, :ξ, :Λ)}(), marginals, nothing) ≈ 6.721945550750932
            end
        end

        begin
            q_out = PointMass([0.3873049736301686, 0.6250550344669505, 0.9865681758036855])
            μ     = [0.21393068431529905, 0.5271266845217295, 0.0032162989237940476]
            Λ     = [1.1178940049080035 1.004643185289189 0.5726053202206195; 1.004643185289189 0.9113807582278761 0.515956570765438; 0.5726053202206195 0.515956570765438 0.3402071079911484]
            q_ξ   = MvNormalWeightedMeanPrecision(Λ * μ, Λ)
            q_Λ   = Wishart(4, [0.4720138588203935 0.49760446060666863 0.4405032929933653; 0.49760446060666863 0.6757755009955487 0.636955076077564; 0.4405032929933653 0.636955076077564 0.6660144475917664])

            for N in (MvNormalMeanPrecision, MvNormalMeanCovariance, MvNormalWeightedMeanPrecision), G in (Wishart,)
                marginals = (Marginal(q_out, false, false, nothing), Marginal(convert(N, q_ξ), false, false, nothing), Marginal(convert(G, q_Λ), false, false, nothing))
                @test score(AverageEnergy(), MvNormalWeightedMeanPrecision, Val{(:out, :ξ, :Λ)}(), marginals, nothing) ≈ 57.362404928342
            end
        end

        begin
            μ_out = [0.8625589157256603, 0.6694783342639599]
            Λ_out = [1.0014322413749484 0.7989099036521625; 0.7989099036521625 1.0976639268696966]
            μ_ξ   = [0.9334416739853251, 0.38318522701093105]
            Λ_ξ   = [0.21867945696266933 0.5704895781120056; 0.5704895781120056 1.5321190185800933]
            q_out = MvNormalWeightedMeanPrecision(Λ_out * μ_out, Λ_out)
            q_ξ   = MvNormalWeightedMeanPrecision(Λ_ξ * μ_ξ, Λ_ξ)
            q_Λ   = Wishart(3, [1.2509510680086597 1.3662166757006484; 1.3662166757006484 1.4925451500802907])

            for N1 in (MvNormalMeanPrecision, MvNormalMeanCovariance, MvNormalWeightedMeanPrecision),
                N2 in (MvNormalMeanPrecision, MvNormalMeanCovariance, MvNormalWeightedMeanPrecision),
                G in (Wishart,)

                marginals = (
                    Marginal(convert(N1, q_out), false, false, nothing), Marginal(convert(N2, q_ξ), false, false, nothing), Marginal(convert(G, q_Λ), false, false, nothing)
                )
                @test score(AverageEnergy(), MvNormalWeightedMeanPrecision, Val{(:out, :ξ, :Λ)}(), marginals, nothing) ≈ 114.57678973411974
            end
        end

        begin
            μ_out_ξ = [0.35156676223859784, 0.6798203100143094, 0.954485919235333, 0.9236981452828203]
            Λ_out_ξ = [
                1.3182839156957349 0.9159049032047119 1.170482409249098 1.132202025059748
                0.9159049032047119 1.4737964254194567 1.4024254322343757 0.7350293025705011
                1.170482409249098 1.4024254322343757 2.0577570913647105 1.3137472032115916
                1.132202025059748 0.7350293025705011 1.3137472032115916 1.2083880803032556
            ]
            q_out_ξ = MvNormalWeightedMeanPrecision(Λ_out_ξ * μ_out_ξ, Λ_out_ξ)
            q_Λ = PointMass([0.6202203324986401 0.6037236520027125; 0.6037236520027125 1.140352058799863])

            for N in (MvNormalMeanPrecision, MvNormalMeanCovariance, MvNormalWeightedMeanPrecision)
                marginals = (Marginal(convert(N, q_out_ξ), false, false, nothing), Marginal(q_Λ, false, false, nothing))
                @test score(AverageEnergy(), MvNormalWeightedMeanPrecision, Val{(:out_ξ, :Λ)}(), marginals, nothing) ≈ 11.240486862933556
            end
        end

        begin
            μ_out_μ = [0.35156676223859784, 0.6798203100143094, 0.954485919235333, 0.9236981452828203]
            Λ_out_μ = [
                1.3182839156957349 0.9159049032047119 1.170482409249098 1.132202025059748
                0.9159049032047119 1.4737964254194567 1.4024254322343757 0.7350293025705011
                1.170482409249098 1.4024254322343757 2.0577570913647105 1.3137472032115916
                1.132202025059748 0.7350293025705011 1.3137472032115916 1.2083880803032556
            ]
            q_out_μ = MvNormalWeightedMeanPrecision(Λ_out_μ * μ_out_μ, Λ_out_μ)
            q_Λ = Wishart(3, [0.6202203324986401 0.6037236520027125; 0.6037236520027125 1.140352058799863])

            for N in (MvNormalMeanPrecision, MvNormalMeanCovariance, MvNormalWeightedMeanPrecision), G in (Wishart,)
                marginals = (Marginal(convert(N, q_out_μ), false, false, nothing), Marginal(convert(G, q_Λ), false, false, nothing))
                @test score(AverageEnergy(), MvNormalWeightedMeanPrecision, Val{(:out_ξ, :Λ)}(), marginals, nothing) ≈ 28.552276936591902
            end
        end
    end
end
end
