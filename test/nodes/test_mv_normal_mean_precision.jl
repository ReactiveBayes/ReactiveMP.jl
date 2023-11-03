module MvNormalMeanPrecisionNodeTest

using Test, ReactiveMP, Random, BayesBase, ExponentialFamily

import ReactiveMP: make_node

@testset "MvNormalMeanPrecisionNode" begin
    @testset "Creation" begin
        node = make_node(MvNormalMeanPrecision)

        @test functionalform(node) === MvNormalMeanPrecision
        @test sdtype(node) === Stochastic()
        @test name.(interfaces(node)) === (:out, :μ, :Λ)
        @test factorisation(node) === ((1, 2, 3),)
        @test localmarginalnames(node) === (:out_μ_Λ,)
        @test metadata(node) === nothing

        node = make_node(MvNormalMeanPrecision, FactorNodeCreationOptions(nothing, 1, nothing))

        @test functionalform(node) === MvNormalMeanPrecision
        @test sdtype(node) === Stochastic()
        @test name.(interfaces(node)) === (:out, :μ, :Λ)
        @test factorisation(node) === ((1, 2, 3),)
        @test localmarginalnames(node) === (:out_μ_Λ,)
        @test metadata(node) === 1

        node = make_node(MvNormalMeanPrecision, FactorNodeCreationOptions(((1,), (2, 3)), nothing, nothing))

        @test functionalform(node) === MvNormalMeanPrecision
        @test sdtype(node) === Stochastic()
        @test name.(interfaces(node)) === (:out, :μ, :Λ)
        @test factorisation(node) === ((1,), (2, 3))
        @test localmarginalnames(node) === (:out, :μ_Λ)
        @test metadata(node) === nothing
    end

    @testset "AverageEnergy" begin
        begin
            q_out = PointMass([1.0, 1.0])
            q_μ   = MvNormalMeanPrecision([1.0, 1.0], [1.0 0.0; 0.0 1.0])
            q_Λ   = Wishart(3, [2.0 0.0; 0.0 2.0])

            for N in (MvNormalMeanPrecision, MvNormalMeanCovariance, MvNormalWeightedMeanPrecision), G in (Wishart,)
                marginals = (Marginal(q_out, false, false, nothing), Marginal(convert(N, q_μ), false, false, nothing), Marginal(convert(G, q_Λ), false, false, nothing))
                @test score(AverageEnergy(), MvNormalMeanPrecision, Val{(:out, :μ, :Λ)}(), marginals, nothing) ≈ 6.721945550750932
            end
        end

        begin
            q_out = PointMass([0.3873049736301686, 0.6250550344669505, 0.9865681758036855])
            q_μ   = MvNormalMeanPrecision([0.21393068431529905, 0.5271266845217295, 0.0032162989237940476], [1.1178940049080035 1.004643185289189 0.5726053202206195; 1.004643185289189 0.9113807582278761 0.515956570765438; 0.5726053202206195 0.515956570765438 0.3402071079911484])
            q_Λ   = Wishart(4, [0.4720138588203935 0.49760446060666863 0.4405032929933653; 0.49760446060666863 0.6757755009955487 0.636955076077564; 0.4405032929933653 0.636955076077564 0.6660144475917664])

            for N in (MvNormalMeanPrecision, MvNormalMeanCovariance, MvNormalWeightedMeanPrecision), G in (Wishart,)
                marginals = (Marginal(q_out, false, false, nothing), Marginal(convert(N, q_μ), false, false, nothing), Marginal(convert(G, q_Λ), false, false, nothing))
                @test score(AverageEnergy(), MvNormalMeanPrecision, Val{(:out, :μ, :Λ)}(), marginals, nothing) ≈ 57.362404928342
            end
        end

        begin
            q_out = MvNormalMeanPrecision([0.8625589157256603, 0.6694783342639599], [1.0014322413749484 0.7989099036521625; 0.7989099036521625 1.0976639268696966])
            q_μ   = MvNormalMeanPrecision([0.9334416739853251, 0.38318522701093105], [0.21867945696266933 0.5704895781120056; 0.5704895781120056 1.5321190185800933])
            q_Λ   = Wishart(3, [1.2509510680086597 1.3662166757006484; 1.3662166757006484 1.4925451500802907])

            for N1 in (MvNormalMeanPrecision, MvNormalMeanCovariance, MvNormalWeightedMeanPrecision),
                N2 in (MvNormalMeanPrecision, MvNormalMeanCovariance, MvNormalWeightedMeanPrecision),
                G in (Wishart,)

                marginals = (
                    Marginal(convert(N1, q_out), false, false, nothing), Marginal(convert(N2, q_μ), false, false, nothing), Marginal(convert(G, q_Λ), false, false, nothing)
                )
                @test score(AverageEnergy(), MvNormalMeanPrecision, Val{(:out, :μ, :Λ)}(), marginals, nothing) ≈ 114.57678973411974
            end
        end

        begin
            q_out = MvNormalMeanPrecision([0.9580573744420284, 0.26086767384401943, 0.28770712080127914], [0.9182022104547379 0.6106281139047683 1.0779505969017997; 0.6106281139047683 1.032830918722213 0.9771172696798327; 1.0779505969017997 0.9771172696798327 1.5109842607623276])
            q_μ   = MvNormalMeanPrecision([0.7234472062526713, 0.2824687847180598, 0.2746840635757333], [0.3590359808189558 0.3331759830932437 0.7133660103700628; 0.3331759830932437 0.9384711630986695 0.972435803354717; 0.7133660103700628 0.972435803354717 1.8219531599263405])
            q_Λ   = Wishart(4, [0.32860287160871043 0.11075216999194112 0.19071629144893898; 0.11075216999194112 0.15595676288444096 0.1700073984759777; 0.19071629144893898 0.1700073984759777 0.2755688590223264])

            for N1 in (MvNormalMeanPrecision, MvNormalMeanCovariance, MvNormalWeightedMeanPrecision),
                N2 in (MvNormalMeanPrecision, MvNormalMeanCovariance, MvNormalWeightedMeanPrecision),
                G in (Wishart,)

                marginals = (
                    Marginal(convert(N1, q_out), false, false, nothing), Marginal(convert(N2, q_μ), false, false, nothing), Marginal(convert(G, q_Λ), false, false, nothing)
                )
                @test score(AverageEnergy(), MvNormalMeanPrecision, Val{(:out, :μ, :Λ)}(), marginals, nothing) ≈ 14.92006071729396
            end
        end

        begin
            q_out = MvNormalMeanPrecision([0.8436742420779029, 0.10538074240979411, 0.6165907670265702], [1.9016529880577202 0.10379920228081287 1.8123623785462684; 0.10379920228081287 0.608931528715827 0.31789777698874117; 1.8123623785462684 0.31789777698874117 1.812858220779861])
            q_μ   = MvNormalMeanPrecision([0.2069541691083061, 0.5589468611740172, 0.4820571901630184], [0.4530148040808207 0.3039331260702957 0.35941114218805115; 0.3039331260702957 0.4389148552241666 0.23999683261779386; 0.35941114218805115 0.23999683261779386 0.3727021595284692])
            q_Λ   = PointMass([0.705864402777104 1.0323593788454744 0.9684301882026219; 1.0323593788454744 1.7349617392942005 1.7059080462235048; 0.9684301882026219 1.7059080462235048 1.816579920226959])

            for N1 in (MvNormalMeanPrecision, MvNormalMeanCovariance, MvNormalWeightedMeanPrecision),
                N2 in (MvNormalMeanPrecision, MvNormalMeanCovariance, MvNormalWeightedMeanPrecision)

                marginals = (Marginal(convert(N1, q_out), false, false, nothing), Marginal(convert(N2, q_μ), false, false, nothing), Marginal(q_Λ, false, false, nothing))
                @test score(AverageEnergy(), MvNormalMeanPrecision, Val{(:out, :μ, :Λ)}(), marginals, nothing) ≈ 38.31309509463591
            end
        end

        begin
            q_out_μ = MvNormalMeanPrecision(
                [0.2932046487282065, 0.7716085147100042, 0.03978072440454361, 0.2814883836121471],
                [
                    1.0684808331872628 0.6721958601342372 0.7164104160110533 0.2869444930570181
                    0.6721958601342372 1.4472786772965438 1.3516272546828385 0.5533932426057602
                    0.7164104160110533 1.3516272546828385 1.2958919150214063 0.5171784879755998
                    0.2869444930570181 0.5533932426057602 0.5171784879755998 0.30898006058959576
                ]
            )
            q_Λ = PointMass([0.6531636197654624 0.8023009003361893; 0.8023009003361893 1.0001231468452463])

            for N in (MvNormalMeanPrecision, MvNormalMeanCovariance, MvNormalWeightedMeanPrecision)
                marginals = (Marginal(convert(N, q_out_μ), false, false, nothing), Marginal(q_Λ, false, false, nothing))
                @test score(AverageEnergy(), MvNormalMeanPrecision, Val{(:out_μ, :Λ)}(), marginals, nothing) ≈ 87.02279699224562
            end
        end

        begin
            q_out_μ = MvNormalMeanPrecision(
                [0.35156676223859784, 0.6798203100143094, 0.954485919235333, 0.9236981452828203],
                [
                    1.3182839156957349 0.9159049032047119 1.170482409249098 1.132202025059748
                    0.9159049032047119 1.4737964254194567 1.4024254322343757 0.7350293025705011
                    1.170482409249098 1.4024254322343757 2.0577570913647105 1.3137472032115916
                    1.132202025059748 0.7350293025705011 1.3137472032115916 1.2083880803032556
                ]
            )
            q_Λ = PointMass([0.6202203324986401 0.6037236520027125; 0.6037236520027125 1.140352058799863])

            for N in (MvNormalMeanPrecision, MvNormalMeanCovariance, MvNormalWeightedMeanPrecision)
                marginals = (Marginal(convert(N, q_out_μ), false, false, nothing), Marginal(q_Λ, false, false, nothing))
                @test score(AverageEnergy(), MvNormalMeanPrecision, Val{(:out_μ, :Λ)}(), marginals, nothing) ≈ 11.240486862933556
            end
        end

        begin
            q_out_μ = MvNormalMeanPrecision(
                [0.35156676223859784, 0.6798203100143094, 0.954485919235333, 0.9236981452828203],
                [
                    1.3182839156957349 0.9159049032047119 1.170482409249098 1.132202025059748
                    0.9159049032047119 1.4737964254194567 1.4024254322343757 0.7350293025705011
                    1.170482409249098 1.4024254322343757 2.0577570913647105 1.3137472032115916
                    1.132202025059748 0.7350293025705011 1.3137472032115916 1.2083880803032556
                ]
            )
            q_Λ = Wishart(3, [0.6202203324986401 0.6037236520027125; 0.6037236520027125 1.140352058799863])

            for N in (MvNormalMeanPrecision, MvNormalMeanCovariance, MvNormalWeightedMeanPrecision), G in (Wishart,)
                marginals = (Marginal(convert(N, q_out_μ), false, false, nothing), Marginal(convert(G, q_Λ), false, false, nothing))
                @test score(AverageEnergy(), MvNormalMeanPrecision, Val{(:out_μ, :Λ)}(), marginals, nothing) ≈ 28.552276936591902
            end
        end

        begin
            q_out_μ = MvNormalMeanPrecision(
                [0.5500318675144917, 0.817183383468487, 0.5347013723720735, 0.5114047207869656],
                [
                    1.1343108996797893 0.47265813998866796 0.9272779005094741 1.0428711245585718
                    0.47265813998866796 0.715974520938224 0.7635970006454071 1.193450872579602
                    0.9272779005094741 0.7635970006454071 1.392689362184066 1.5818118006093156
                    1.0428711245585718 1.193450872579602 1.5818118006093156 2.158417891827849
                ]
            )
            q_Λ = Wishart(3, [1.545789934380036 1.1984283177361577; 1.1984283177361577 1.1061038557496674])

            for N in (MvNormalMeanPrecision, MvNormalMeanCovariance, MvNormalWeightedMeanPrecision), G in (Wishart,)
                marginals = (Marginal(convert(N, q_out_μ), false, false, nothing), Marginal(convert(G, q_Λ), false, false, nothing))
                @test score(AverageEnergy(), MvNormalMeanPrecision, Val{(:out_μ, :Λ)}(), marginals, nothing) ≈ 1938.6631975673586
            end
        end
    end
end
end
