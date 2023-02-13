module MvNormalMeanCovarianceNodeTest

using Test
using ReactiveMP
using Random

import ReactiveMP: make_node

@testset "MvNormalMeanCovarianceNode" begin
    @testset "Creation" begin
        node = make_node(MvNormalMeanCovariance)

        @test functionalform(node) === MvNormalMeanCovariance
        @test sdtype(node) === Stochastic()
        @test name.(interfaces(node)) === (:out, :μ, :Σ)
        @test factorisation(node) === ((1, 2, 3),)
        @test localmarginalnames(node) === (:out_μ_Σ,)
        @test metadata(node) === nothing

        node = make_node(MvNormalMeanCovariance, FactorNodeCreationOptions(nothing, 1, nothing))

        @test functionalform(node) === MvNormalMeanCovariance
        @test sdtype(node) === Stochastic()
        @test name.(interfaces(node)) === (:out, :μ, :Σ)
        @test factorisation(node) === ((1, 2, 3),)
        @test localmarginalnames(node) === (:out_μ_Σ,)
        @test metadata(node) === 1

        node = make_node(MvNormalMeanCovariance, FactorNodeCreationOptions(((1,), (2, 3)), nothing, nothing))

        @test functionalform(node) === MvNormalMeanCovariance
        @test sdtype(node) === Stochastic()
        @test name.(interfaces(node)) === (:out, :μ, :Σ)
        @test factorisation(node) === ((1,), (2, 3))
        @test localmarginalnames(node) === (:out, :μ_Σ)
        @test metadata(node) === nothing
    end

    @testset "AverageEnergy" begin
        begin
            q_out = PointMass([1.0, 1.0])
            q_μ   = MvNormalMeanCovariance([1.0, 1.0], [1.0 0.0; 0.0 1.0])
            q_Σ   = PointMass([2.0 0.0; 0.0 2.0])

            for N in (MvNormalMeanPrecision, MvNormalMeanCovariance, MvNormalWeightedMeanPrecision)
                marginals = (Marginal(q_out, false, false, nothing), Marginal(convert(N, q_μ), false, false, nothing), Marginal(q_Σ, false, false, nothing))
                @test score(AverageEnergy(), MvNormalMeanCovariance, Val{(:out, :μ, :Σ)}, marginals, nothing) ≈ 3.0310242469692907
            end
        end

        begin
            q_out = MvNormalMeanCovariance([0.34777017741128646, 0.478300212208703], [0.36248838736696753 0.42718825177834396; 0.42718825177834396 0.6613345138178703])
            q_μ   = MvNormalMeanCovariance([0.28114649146383175, 0.7773140083754762], [0.5753219837395999 0.57742814389364; 0.57742814389364 0.7164979237943061])
            q_Σ   = PointMass([0.9575163948991356 0.952247049670764; 0.952247049670764 1.001998675194526])

            for N1 in (MvNormalMeanPrecision, MvNormalMeanCovariance, MvNormalWeightedMeanPrecision),
                N2 in (MvNormalMeanPrecision, MvNormalMeanCovariance, MvNormalWeightedMeanPrecision)

                marginals = (Marginal(convert(N1, q_out), false, false, nothing), Marginal(convert(N2, q_μ), false, false, nothing), Marginal(q_Σ, false, false, nothing))
                @test score(AverageEnergy(), MvNormalMeanCovariance, Val{(:out, :μ, :Σ)}, marginals, nothing) ≈ 4.863921806562866
            end
        end

        begin
            q_out = MvNormalMeanCovariance([0.3685447210709054, 0.7059290804186025], [0.47091887707487257 0.1715872501382935; 0.1715872501382935 0.07937045688177925])
            q_μ   = MvNormalMeanCovariance([0.9821121935372943, 0.8740184196009864], [1.2740987247403437 0.376849163380472; 0.376849163380472 0.17831073369063927])
            q_Σ   = PointMass([0.761731832494188 0.21222845781825198; 0.21222845781825198 0.5989429144303828])

            for N1 in (MvNormalMeanPrecision, MvNormalMeanCovariance, MvNormalWeightedMeanPrecision),
                N2 in (MvNormalMeanPrecision, MvNormalMeanCovariance, MvNormalWeightedMeanPrecision)

                marginals = (Marginal(convert(N1, q_out), false, false, nothing), Marginal(convert(N2, q_μ), false, false, nothing), Marginal(q_Σ, false, false, nothing))
                @test score(AverageEnergy(), MvNormalMeanCovariance, Val{(:out, :μ, :Σ)}, marginals, nothing) ≈ 2.867156770180465
            end
        end

        begin
            q_out_μ = MvNormalMeanCovariance(
                [0.7110546689402384, 0.2550803989358359, 0.7284714504680616, 0.4523569391382931],
                [
                    0.9354883429919909 0.979541773915805 0.11235208550731414 0.720928227797172
                    0.979541773915805 1.5518339076149474 0.27953313117565737 0.9410472233142694
                    0.11235208550731414 0.27953313117565737 0.09215639567249542 0.15210811270662963
                    0.720928227797172 0.9410472233142694 0.15210811270662963 0.7114840285830272
                ]
            )
            q_Σ = PointMass([0.6588719526947034 0.3784788140113857; 0.3784788140113857 0.2611292983123214])

            for N in (MvNormalMeanPrecision, MvNormalMeanCovariance, MvNormalWeightedMeanPrecision)
                marginals = (Marginal(convert(N, q_out_μ), false, false, nothing), Marginal(q_Σ, false, false, nothing))
                @test score(AverageEnergy(), MvNormalMeanCovariance, Val{(:out_μ, :Σ)}, marginals, nothing) ≈ 6.741420950973408
            end
        end

        begin
            q_out_μ = MvNormalMeanCovariance(
                [0.6273836373763129, 0.2611455272882084, 0.15015693447418932, 0.7907870618820343],
                [
                    2.4624395254748563 1.3049173253859347 2.2581875619221057 1.8418052081652991
                    1.3049173253859347 1.0230726719658598 1.0344117375112738 0.8684398658701927
                    2.2581875619221057 1.0344117375112738 2.164616392026885 1.7042534061619456
                    1.8418052081652991 0.8684398658701927 1.7042534061619456 1.5840784999702135
                ]
            )
            q_Σ = PointMass([0.9918745791319346 0.7102392994361006; 0.7102392994361006 0.5209829748135837])

            for N in (MvNormalMeanPrecision, MvNormalMeanCovariance, MvNormalWeightedMeanPrecision)
                marginals = (Marginal(convert(N, q_out_μ), false, false, nothing), Marginal(q_Σ, false, false, nothing))
                @test score(AverageEnergy(), MvNormalMeanCovariance, Val{(:out_μ, :Σ)}, marginals, nothing) ≈ 60.075730792149585
            end
        end
    end
end
end
