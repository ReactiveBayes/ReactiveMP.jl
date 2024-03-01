
@testitem "MvNormalMeanScalePrecision" begin
    using ReactiveMP, Random, BayesBase, ExponentialFamily

    import ReactiveMP: make_node

    @testset "Creation" begin
        node = make_node(MvNormalMeanScalePrecision)

        @test functionalform(node) === MvNormalMeanScalePrecision
        @test sdtype(node) === Stochastic()
        @test name.(interfaces(node)) === (:out, :μ, :γ)
        @test factorisation(node) === ((1, 2, 3),)
        @test localmarginalnames(node) === (:out_μ_γ,)
        @test metadata(node) === nothing

        node = make_node(MvNormalMeanScalePrecision, FactorNodeCreationOptions(nothing, 1, nothing))

        @test functionalform(node) === MvNormalMeanScalePrecision
        @test sdtype(node) === Stochastic()
        @test name.(interfaces(node)) === (:out, :μ, :γ)
        @test factorisation(node) === ((1, 2, 3),)
        @test localmarginalnames(node) === (:out_μ_γ,)
        @test metadata(node) === 1

        node = make_node(MvNormalMeanScalePrecision, FactorNodeCreationOptions(((1,), (2, 3)), nothing, nothing))

        @test functionalform(node) === MvNormalMeanScalePrecision
        @test sdtype(node) === Stochastic()
        @test name.(interfaces(node)) === (:out, :μ, :γ)
        @test factorisation(node) === ((1,), (2, 3))
        @test localmarginalnames(node) === (:out, :μ_γ)
        @test metadata(node) === nothing
    end

    @testset "AverageEnergy" begin
        begin
            q_out = PointMass([1.0, 1.0])
            q_μ   = MvNormalMeanPrecision([1.0, 1.0], [1.0 0.0; 0.0 1.0])
            q_γ   = GammaShapeRate(1.0, 1.0)

            for N in (MvNormalMeanPrecision, MvNormalMeanCovariance, MvNormalWeightedMeanPrecision), g in (Gamma,)
                marginals = (Marginal(q_out, false, false, nothing), Marginal(convert(N, q_μ), false, false, nothing), Marginal(convert(g, q_γ), false, false, nothing))
                @test score(AverageEnergy(), MvNormalMeanScalePrecision, Val{(:out, :μ, :γ)}(), marginals, nothing) ≈ 3.415092731310877
            end
        end

        begin
            q_out = MvNormalMeanPrecision([0.8625589157256603, 0.6694783342639599], [1.0014322413749484 0.7989099036521625; 0.7989099036521625 1.0976639268696966])
            q_μ   = MvNormalMeanPrecision([0.9334416739853251, 0.38318522701093105], [0.21867945696266933 0.5704895781120056; 0.5704895781120056 1.5321190185800933])
            q_γ   = Gamma(2.0, 1.0)

            for N1 in (MvNormalMeanPrecision, MvNormalMeanCovariance, MvNormalWeightedMeanPrecision),
                N2 in (MvNormalMeanPrecision, MvNormalMeanCovariance, MvNormalWeightedMeanPrecision),
                g in (Gamma,)

                marginals = (
                    Marginal(convert(N1, q_out), false, false, nothing), Marginal(convert(N2, q_μ), false, false, nothing), Marginal(convert(g, q_γ), false, false, nothing)
                )
                @test score(AverageEnergy(), MvNormalMeanScalePrecision, Val{(:out, :μ, :γ)}(), marginals, nothing) ≈ 188.7235844555108
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
            q_γ = GammaShapeRate(3.0, 2.0)

            for N in (MvNormalMeanPrecision, MvNormalMeanCovariance, MvNormalWeightedMeanPrecision)
                marginals = (Marginal(convert(N, q_out_μ), false, false, nothing), Marginal(q_γ, false, false, nothing))
                @test score(AverageEnergy(), MvNormalMeanScalePrecision, Val{(:out_μ, :γ)}(), marginals, nothing) ≈ 86.41549408285206
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
            q_γ = GammaShapeRate(4.0, 3.0)

            for N in (MvNormalMeanPrecision, MvNormalMeanCovariance, MvNormalWeightedMeanPrecision)
                marginals = (Marginal(convert(N, q_out_μ), false, false, nothing), Marginal(q_γ, false, false, nothing))
                @test score(AverageEnergy(), MvNormalMeanScalePrecision, Val{(:out_μ, :γ)}(), marginals, nothing) ≈ 11.887416710256351
            end
        end
    end
end
