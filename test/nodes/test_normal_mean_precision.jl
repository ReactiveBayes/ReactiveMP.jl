module NormalMeanPrecisionNodeTest

using Test
using ReactiveMP
using Random

import ReactiveMP: make_node

@testset "NormalMeanPrecisionNode" begin

    @testset "Creation" begin
        node = make_node(NormalMeanPrecision)

        @test functionalform(node)          === NormalMeanPrecision
        @test sdtype(node)                  === Stochastic()
        @test name.(interfaces(node))       === (:out, :μ, :τ)
        @test factorisation(node)           === ((1, 2, 3), )
        @test localmarginalnames(node)      === (:out_μ_τ, )
        @test metadata(node)                === nothing

        node = make_node(NormalMeanPrecision, meta = 1)

        @test functionalform(node)          === NormalMeanPrecision
        @test sdtype(node)                  === Stochastic()
        @test name.(interfaces(node))       === (:out, :μ, :τ)
        @test factorisation(node)           === ((1, 2, 3), )
        @test localmarginalnames(node)      === (:out_μ_τ, )
        @test metadata(node)                === 1

        node = make_node(NormalMeanPrecision, factorisation = ((1, ), (2, 3)))

        @test functionalform(node)          === NormalMeanPrecision
        @test sdtype(node)                  === Stochastic()
        @test name.(interfaces(node))       === (:out, :μ, :τ)
        @test factorisation(node)           === ((1, ), (2, 3, ))
        @test localmarginalnames(node)      === (:out, :μ_τ, )
        @test metadata(node)                === nothing
    end

    @testset "AverageEnergy" begin 

        begin 
            q_out = PointMass(1.0)
            q_μ   = NormalMeanPrecision(1.0, 1.0)
            q_τ   = GammaShapeRate(1.5, 1.5)

            for N in (NormalMeanPrecision, NormalMeanVariance, NormalWeightedMeanPrecision), G in (GammaShapeRate, GammaShapeScale)
                marginals = (Marginal(q_out, false, false), Marginal(convert(N, q_μ), false, false), Marginal(convert(G, q_τ), false, false))
                @test score(AverageEnergy(), NormalMeanPrecision, Val{(:out, :μ, :τ)}, marginals, nothing) ≈ 1.6034261002694663
            end
        end

        begin 
            q_out = NormalMeanPrecision(1.0, 1.0)
            q_μ   = NormalMeanPrecision(1.0, 1.0)
            q_τ   = GammaShapeRate(1.5, 1.5)

            for N in (NormalMeanPrecision, NormalMeanVariance, NormalWeightedMeanPrecision), G in (GammaShapeRate, GammaShapeScale)
                marginals = (Marginal(q_out, false, false), Marginal(convert(N, q_μ), false, false), Marginal(convert(G, q_τ), false, false))
                @test score(AverageEnergy(), NormalMeanPrecision, Val{(:out, :μ, :τ)}, marginals, nothing) ≈ 2.1034261002694663
            end
        end

        begin 
            q_out = PointMass(0.956629)
            q_μ   = NormalMeanPrecision(0.255332, 0.762870)
            q_τ   = GammaShapeRate(0.93037, 0.79312)

            for N in (NormalMeanPrecision, NormalMeanVariance, NormalWeightedMeanPrecision), G in (GammaShapeRate, GammaShapeScale)
                marginals = (Marginal(q_out, false, false), Marginal(convert(N, q_μ), false, false), Marginal(convert(G, q_τ), false, false))
                @test score(AverageEnergy(), NormalMeanPrecision, Val{(:out, :μ, :τ)}, marginals, nothing) ≈ 2.209338084063204
            end
        end

        begin 
            q_out = NormalMeanPrecision(0.148725, 0.483501)
            q_μ   = NormalMeanPrecision(0.992776, 0.545851)
            q_τ   = GammaShapeRate(0.309396, 0.343814)

            for N in (NormalMeanPrecision, NormalMeanVariance, NormalWeightedMeanPrecision), G in (GammaShapeRate, GammaShapeScale)
                marginals = (Marginal(q_out, false, false), Marginal(convert(N, q_μ), false, false), Marginal(convert(G, q_τ), false, false))
                @test score(AverageEnergy(), NormalMeanPrecision, Val{(:out, :μ, :τ)}, marginals, nothing) ≈ 4.155913074139921
            end
        end

        begin 
            q_out_μ = MvNormalMeanPrecision([0.2818402997601115, 0.0847764277628964], [0.7059042678955475 0.3595204552322394; 0.3595204552322394 0.22068491824258746])
            q_τ     = GammaShapeRate(0.49074414, 0.4071772)

            for N in (MvNormalMeanPrecision, MvNormalMeanCovariance, MvNormalWeightedMeanPrecision), G in (GammaShapeRate, GammaShapeScale)
                marginals = (Marginal(convert(N, q_out_μ), false, false), Marginal(convert(G, q_τ), false, false))
                @test score(AverageEnergy(), NormalMeanPrecision, Val{(:out_μ, :τ)}, marginals, nothing) ≈ 38.88138702883774
            end
        end

        begin 
            q_out_μ = MvNormalMeanPrecision([0.8378350736808462, 0.41494396892699026], [1.0074451742986652 0.4369298270351709; 0.4369298270351709 0.19572138039218784])
            q_τ     = GammaShapeRate(0.435485, 0.5269575)

            for N in (MvNormalMeanPrecision, MvNormalMeanCovariance, MvNormalWeightedMeanPrecision), G in (GammaShapeRate, GammaShapeScale)
                marginals = (Marginal(convert(N, q_out_μ), false, false), Marginal(convert(G, q_τ), false, false))
                @test score(AverageEnergy(), NormalMeanPrecision, Val{(:out_μ, :τ)}, marginals, nothing) ≈ 138.6947657738283
            end
        end

    end

end
end
