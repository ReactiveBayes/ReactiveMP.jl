module ReactiveMPRenderCVITest

using Test
using ReactiveMP
using Random
using StableRNGs
using Distributions
using Zygote
using Flux
using DiffResults

import ReactiveMP: naturalparams, NaturalParameters, AbstractContinuousGenericLogPdf

struct EmptyOptimizer end

function ReactiveMP.cvi_update!(::EmptyOptimizer, λ, ∇)
    return vec(λ)
end

mutable struct CountingOptimizer
    num_its::Int
end

function ReactiveMP.cvi_update!(opt::CountingOptimizer, λ, ∇)
    opt.num_its += 1
    return vec(λ)
end

@testset "cvi:prod" begin
    @testset "empty optimizer" begin
        for (m_in, m_out) in ((NormalMeanVariance(0, 1), NormalMeanVariance(0, 1)), (GammaShapeRate(2, 2), GammaShapeRate(2, 2)), (Bernoulli(0.5), Bernoulli(0.5)))
            opt = EmptyOptimizer()
            meta = CVI(1, 100, opt)
            λ = prod(meta, ContinuousUnivariateLogPdf((z) -> logpdf(m_out, z)), m_in)
            @test all(mean_var(λ) .== mean_var(m_in))
        end
    end

    @testset "counting optimizer" begin
        for (m_in, m_out) in ((NormalMeanVariance(0, 1), NormalMeanVariance(0, 1)), (GammaShapeRate(2, 2), GammaShapeRate(2, 2)), (Bernoulli(0.5), Bernoulli(0.5)))
            opt = CountingOptimizer(0)
            meta = CVI(1, 100, opt)
            λ = prod(meta, ContinuousUnivariateLogPdf((z) -> logpdf(m_out, z)), m_in)
            @test all(mean_var(λ) .== mean_var(m_in))
            @test opt.num_its === 100
        end
    end

    @testset "counting lambda optimizer" begin
        for (m_in, m_out) in ((NormalMeanVariance(0, 1), NormalMeanVariance(0, 1)), (GammaShapeRate(2, 2), GammaShapeRate(2, 2)), (Bernoulli(0.5), Bernoulli(0.5)))
            η = naturalparams(m_in)
            num_its = 0
            opt = (λ, ∇) -> begin
                num_its += 1
                return vec(λ)
            end
            meta = CVI(1, 100, opt)
            λ = prod(meta, ContinuousUnivariateLogPdf((z) -> logpdf(m_out, z)), m_in)
            @test all(mean_var(λ) .== mean_var(m_in))
            @test num_its === 100
        end
    end

    @testset "counting lambda optimizer (Zygote Grad)" begin
        for (m_in, m_out) in ((NormalMeanVariance(0, 1), NormalMeanVariance(0, 1)), (GammaShapeRate(2, 2), GammaShapeRate(2, 2)), (Bernoulli(0.5), Bernoulli(0.5)))
            num_its = 0
            opt = (λ, ∇) -> begin
                num_its += 1
                return vec(λ)
            end
            meta = CVI(1, 100, opt, ZygoteGrad())
            λ = prod(meta, ContinuousUnivariateLogPdf((z) -> logpdf(m_out, z)), m_in)
            @test all(mean_var(λ) .== mean_var(m_in))
            @test num_its === 100
        end
    end

    @testset "cvi `prod` tests" begin
        rng = StableRNG(42)

        tests = (
            (method = CVI(StableRNG(42), 1, 1000, Descent(0.01), ForwardDiffGrad(), false, true), tol = 5e-1),
            (method = CVI(StableRNG(42), 1, 1000, Descent(0.01), ZygoteGrad(), false, true), tol = 5e-1)
        )

        # Check several prods against their analytical solutions
        for test in tests, i in 1:5

            # Univariate `Normal`
            n1 = NormalMeanVariance(10 * randn(rng), 10 * rand(rng))
            n2 = NormalMeanVariance(10 * randn(rng), 10 * rand(rng))

            n_analytical = prod(ProdAnalytical(), n1, n2)
            n_cvi = prod(test[:method], ContinuousUnivariateLogPdf((x) -> logpdf(n1, x)), n2)

            @test n_analytical ≈ n_cvi atol = test[:tol]

            # Univariate `Gamma`
            g1 = GammaShapeRate(rand(rng) + 1, rand(rng) + 1)
            g2 = GammaShapeRate(rand(rng) + 1, rand(rng) + 1)

            g_analytical = prod(ProdAnalytical(), g1, g2)
            g_cvi = prod(test[:method], ContinuousUnivariateLogPdf((x) -> logpdf(g1, x)), g2)

            @test all(isapprox.(mean_var(g_analytical), mean_var(g_cvi), atol = test[:tol]))

            # Multivariate `Normal`
            if !(ReactiveMP.get_grad(test[:method]) isa ZygoteGrad) # `Zygote` does not support mutations
                for d in (2, 3)
                    mn1 = MvNormalMeanCovariance(10 * randn(rng, d), 10 * rand(rng, d))
                    mn2 = MvNormalMeanCovariance(10 * randn(rng, d), 10 * rand(rng, d))

                    mn_analytical = prod(ProdAnalytical(), mn1, mn2)
                    mn_cvi = prod(test[:method], ContinuousMultivariateLogPdf(d, (x) -> logpdf(mn1, x)), mn2)

                    @test mn_analytical ≈ mn_cvi atol = test[:tol]
                end
            end
        end
    end

    @testset "Normal x Normal (Log-likelihood preconditioner prod)" begin
        seed = 123
        rng = StableRNG(seed)
        optimizer = Descent(0.01)
        meta = CVI(rng, 1, 1000, optimizer, ForwardDiffGrad(), false, false)

        for i in 1:10
            m_out, m_in = NormalMeanVariance(i, 1), NormalMeanVariance(0, 1)
            λ = prod(meta, ContinuousUnivariateLogPdf((z) -> logpdf(m_out, z)), m_in)
            @test isapprox(convert(Distribution, λ), NormalWeightedMeanPrecision(i, 2), atol = 0.1)
        end
    end

    @static if VERSION ≥ v"1.7" # Base.@invoke is available only in Julia >= 1.7
        @testset "Normal x Normal (Fisher preconditioner prod)" begin
            seed = 123
            rng = StableRNG(seed)
            optimizer = Descent(0.001)
            meta = CVI(rng, 1, 5000, optimizer, ForwardDiffGrad(), false, false)

            for i in 1:3
                m_out, m_in = NormalMeanVariance(i, 1), NormalMeanVariance(0, 1)
                λ = Base.@invoke prod(meta::CVI, (ContinuousUnivariateLogPdf((z) -> logpdf(m_out, z)))::AbstractContinuousGenericLogPdf, m_in::Any)
                @test isapprox(convert(Distribution, λ), NormalWeightedMeanPrecision(i, 2), atol = 0.5)
            end
        end
    end
end

end
