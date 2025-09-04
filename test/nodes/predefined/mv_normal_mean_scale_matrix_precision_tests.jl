
@testitem "MvNormalMeanScaleMatrixPrecision" begin
    using ReactiveMP, Random, BayesBase, ExponentialFamily, LinearAlgebra

    @testset "AverageEnergy" begin
        begin
            q_out = PointMass([1.0, 1.0])
            q_μ = MvNormalMeanPrecision([1.0, 1.0], [1.0 0.0; 0.0 1.0])
            q_γ = GammaShapeRate(1.0, 1.0)
            q_G = Wishart(ndims(q_out) + 2, [1.0 0.0; 0.0 1.0])
            # m_μ, Cov_μ = mean_cov(q_μ)
            # m_out, Cov_out = mean_cov(q_out)
            # AE = - div(ndims(q_μ),2) * mean(log,q_γ) - 0.5*mean(logdet, q_G) + div(ndims(q_μ),2)*log(2pi) + 0.5*tr(mean(q_γ)*mean(q_G)*( Cov_out + Cov_μ + (m_out - m_μ)*(m_out - m_μ)'))

            for N in (MvNormalMeanPrecision, MvNormalMeanCovariance, MvNormalWeightedMeanPrecision), g in (Gamma,), M in (Wishart,)
                marginals = (Marginal(q_out, false, false, nothing), Marginal(convert(N, q_μ), false, false, nothing), Marginal(convert(g, q_γ), false, false, nothing), Marginal(convert(M, q_G), false, false, nothing))
                @test score(AverageEnergy(), MvNormalMeanScaleMatrixPrecision, Val{(:out, :μ, :γ, :G)}(), marginals, nothing) ≈ 5.49230839621241
            end
        end

        begin
            q_out = MvNormalMeanPrecision([0.8625589157256603, 0.6694783342639599], [1.0014322413749484 0.7989099036521625; 0.7989099036521625 1.0976639268696966])
            q_μ = MvNormalMeanPrecision([0.9334416739853251, 0.38318522701093105], [0.21867945696266933 0.5704895781120056; 0.5704895781120056 1.5321190185800933])
            q_γ = Gamma(2.0, 1.0)
            q_G = Wishart(ndims(q_out) + 2, 0.25*[1.0 0.0; 0.0 1.0])
            # m_μ, Cov_μ = mean_cov(q_μ)
            # m_out, Cov_out = mean_cov(q_out)
            # AE = - div(ndims(q_μ),2) * mean(log,q_γ) - 0.5*mean(logdet, q_G) + div(ndims(q_μ),2)*log(2pi) + 0.5*tr(mean(q_γ)*mean(q_G)*( Cov_out + Cov_μ + (m_out - m_μ)*(m_out - m_μ)'))

            for N1 in (MvNormalMeanPrecision, MvNormalMeanCovariance, MvNormalWeightedMeanPrecision),
                N2 in (MvNormalMeanPrecision, MvNormalMeanCovariance, MvNormalWeightedMeanPrecision),
                g in (Gamma,),
                M in (Wishart,)

                marginals = (
                    Marginal(convert(N1, q_out), false, false, nothing), Marginal(convert(N2, q_μ), false, false, nothing), Marginal(convert(g, q_γ), false, false, nothing), Marginal(convert(M, q_G), false, false, nothing)
                )
                @test score(AverageEnergy(), MvNormalMeanScaleMatrixPrecision, Val{(:out, :μ, :γ, :G)}(), marginals, nothing) ≈ 189.18709448153223
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
            d = div(ndims(q_out_μ), 2)
            q_γ = GammaShapeRate(3.0, 2.0)
            q_G = Wishart(d + 2, 0.25*[0.349811  0.318591; 0.318591  0.401713])
            # m_out_μ, Cov_out_μ = mean_cov(q_out_μ)
            # m_out, m_μ = @views m_out_μ[1:d], m_out_μ[(d + 1):end]
            # Cov_out, Cov_μ = @views Cov_out_μ[1:d, 1:d], Cov_out_μ[(d + 1):end, (d + 1):end]
            # Cov_out_out, Cov_μ_μ = @views Cov_out_μ[1:d, (d + 1):end], Cov_out_μ[(d + 1):end, 1:d]
            # AE = - div(ndims(q_μ),2) * mean(log,q_γ) - 0.5*mean(logdet, q_G) + div(ndims(q_μ),2)*log(2pi) + 0.5*tr(mean(q_γ)*mean(q_G)*( Cov_out + Cov_μ - Cov_out_out - Cov_μ_μ + (m_out - m_μ)*(m_out - m_μ)'))


            for N in (MvNormalMeanPrecision, MvNormalMeanCovariance, MvNormalWeightedMeanPrecision)
                marginals = (Marginal(convert(N, q_out_μ), false, false, nothing), Marginal(q_γ, false, false, nothing), Marginal(q_G, false, false, nothing))
                @test score(AverageEnergy(), MvNormalMeanScaleMatrixPrecision, Val{(:out_μ, :γ, :G)}(), marginals, nothing) ≈ 57.01394241406646
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
            q_G = Wishart(ndims(q_out) + 2, 0.25*[5.60439  4.34489; 4.34489  3.69273])
            # m_out_μ, Cov_out_μ = mean_cov(q_out_μ)
            # m_out, m_μ = @views m_out_μ[1:d], m_out_μ[(d + 1):end]
            # Cov_out, Cov_μ = @views Cov_out_μ[1:d, 1:d], Cov_out_μ[(d + 1):end, (d + 1):end]
            # Cov_out_out, Cov_μ_μ = @views Cov_out_μ[1:d, (d + 1):end], Cov_out_μ[(d + 1):end, 1:d]
            # AE = - div(ndims(q_μ),2) * mean(log,q_γ) - 0.5*mean(logdet, q_G) + div(ndims(q_μ),2)*log(2pi) + 0.5*tr(mean(q_γ)*mean(q_G)*( Cov_out + Cov_μ - Cov_out_out - Cov_μ_μ + (m_out - m_μ)*(m_out - m_μ)'))

            for N in (MvNormalMeanPrecision, MvNormalMeanCovariance, MvNormalWeightedMeanPrecision)
                marginals = (Marginal(convert(N, q_out_μ), false, false, nothing), Marginal(q_γ, false, false, nothing), Marginal(q_G, false, false, nothing))
                @test score(AverageEnergy(), MvNormalMeanScaleMatrixPrecision, Val{(:out_μ, :γ, :G)}(), marginals, nothing) ≈ 60.58871007449749
            end
        end
    end
end
