module ReactiveMPRenderCVITest

using Test, ReactiveMP, Random, StableRNGs, BayesBase, Distributions, ExponentialFamily, Zygote, Optimisers, DiffResults

import BayesBase: AbstractContinuousGenericLogPdf
import StatsFuns: logistic, softmax
import SpecialFunctions: polygamma

struct EmptyOptimizer end

function ReactiveMP.cvi_setup(::EmptyOptimizer, λ)
    return EmptyOptimizer()
end

function ReactiveMP.cvi_update!(::EmptyOptimizer, λ, ∇)
    return vec(λ)
end

mutable struct CountingOptimizer
    num_its::Int
end

function ReactiveMP.cvi_setup(opt::CountingOptimizer, λ)
    return opt
end

function ReactiveMP.cvi_update!(opt::CountingOptimizer, λ, ∇)
    opt.num_its += 1
    return vec(λ)
end

function gammafisher(dist::GammaShapeRate)
    return [polygamma(1, shape(dist)) -1/rate(dist); -1/rate(dist) shape(dist)/rate(dist)^2]
end

@testset "cvi:prod" begin

    # These are tested in the `ExponentialFamily`, test a simple case just to be sure
    @testset "cvi:informationmatrix" begin
        @testset "GammaShapeRate" begin
            for i in 1:10, j in 1:10
                distribution = GammaShapeRate(i, j)
                expform = convert(ExponentialFamilyDistribution, distribution)
                F = fisherinformation(expform)
                @test [1 0; 0 -1]' * F * [1 0; 0 -1] ≈ gammafisher(distribution)
            end
        end
    end

    @testset "ForwardDiffGrad" begin
        grad = ForwardDiffGrad()
    
        for i in 1:100
            @test ReactiveMP.compute_gradient(grad, (x) -> sum(x)^2, [i]) ≈ [2 * i]
            @test ReactiveMP.compute_hessian(grad, (x) -> sum(x)^2, [i]) ≈ [2;;]
        end
    end

    @testset "empty optimizer" begin
        for (m_in, m_out) in ((NormalMeanVariance(0, 1), NormalMeanVariance(0, 1)), (GammaShapeRate(2, 2), GammaShapeRate(2, 2)), (Bernoulli(0.5), Bernoulli(0.5)))
            opt = EmptyOptimizer()
            meta = CVI(1, 100, opt, ForwardDiffGrad(), 1, Val(false), false)
            λ = prod(meta, ContinuousUnivariateLogPdf((z) -> logpdf(m_out, z)), m_in)
            @test all(mean_var(λ) .== mean_var(m_in))
        end
    end

    @testset "counting optimizer" begin
        for (m_in, m_out) in ((NormalMeanVariance(0, 1), NormalMeanVariance(0, 1)), (GammaShapeRate(2, 2), GammaShapeRate(2, 2)), (Bernoulli(0.5), Bernoulli(0.5)))
            opt = CountingOptimizer(0)
            meta = CVI(1, 100, opt, ForwardDiffGrad(), 1, Val(false), false)
            λ = prod(meta, ContinuousUnivariateLogPdf((z) -> logpdf(m_out, z)), m_in)
            @test all(mean_var(λ) .== mean_var(m_in))
            @test opt.num_its === 100
        end
    end

    @testset "counting lambda optimizer" begin
        for (m_in, m_out) in ((NormalMeanVariance(0, 1), NormalMeanVariance(0, 1)), (GammaShapeRate(2, 2), GammaShapeRate(2, 2)), (Bernoulli(0.5), Bernoulli(0.5)))
            num_its = 0
            opt = (λ, ∇) -> begin
                num_its += 1
                return vec(λ)
            end
            meta = CVI(1, 100, opt, ForwardDiffGrad(), 1, Val(false), false)
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
            meta = CVI(1, 100, opt, ZygoteGrad(), 1, Val(false), false)
            λ = prod(meta, ContinuousUnivariateLogPdf((z) -> logpdf(m_out, z)), m_in)
            @test all(mean_var(λ) .== mean_var(m_in))
            @test num_its === 100
        end
    end

    @testset "cvi `prod` tests" begin
        rng = StableRNG(42)

        tests = (
            (method = CVI(StableRNG(42), 1, 1000, Optimisers.Descent(0.007), ForwardDiffGrad(), 10, Val(true), true), tol = 2e-1),
            (method = CVI(StableRNG(42), 1, 1000, Optimisers.Descent(0.007), ZygoteGrad(), 10, Val(true), true), tol = 2e-1)
        )

        # Check several prods against their analytical solutions
        for test in tests, i in 1:5

            # Univariate `Normal`
            n1 = NormalMeanVariance(10 * randn(rng), 10 * rand(rng))
            n2 = NormalMeanVariance(10 * randn(rng), 10 * rand(rng))

            n_analytical = prod(GenericProd(), n1, n2)

            @test prod(test[:method], ContinuousUnivariateLogPdf((x) -> logpdf(n1, x)), n2) ≈ n_analytical atol = test[:tol]
            @test prod(test[:method], n1, n2) ≈ n_analytical atol = test[:tol]

            # Univariate `Gamma`
            g1 = GammaShapeRate(rand(rng) + 1, rand(rng) + 1)
            g2 = GammaShapeRate(rand(rng) + 1, rand(rng) + 1)

            g_analytical = prod(GenericProd(), g1, g2)
            g_cvi1 = prod(test[:method], g1, g2)
            g_cvi2 = prod(test[:method], ContinuousUnivariateLogPdf((x) -> logpdf(g1, x)), g2)

            @test all(isapprox.(mean_var(g_analytical), mean_var(g_cvi1), atol = test[:tol]))
            @test all(isapprox.(mean_var(g_analytical), mean_var(g_cvi2), atol = test[:tol]))

            # Multivariate `Normal`
            if !(test[:method].grad isa ZygoteGrad) # `Zygote` does not support mutations
                for d in (2, 3)
                    mn1 = MvNormalMeanCovariance(10 * randn(rng, d), 10 * rand(rng, d))
                    mn2 = MvNormalMeanCovariance(10 * randn(rng, d), 10 * rand(rng, d))

                    mn_analytical = prod(GenericProd(), mn1, mn2)

                    @test prod(test[:method], mn1, mn2) ≈ mn_analytical atol = test[:tol]
                    @test prod(test[:method], ContinuousMultivariateLogPdf(d, (x) -> logpdf(mn1, x)), mn2) ≈ mn_analytical atol = test[:tol]
                end
            end

            b1 = Bernoulli(logistic(randn(rng)))
            b2 = Bernoulli(logistic(randn(rng)))
            b_analytical = prod(GenericProd(), b1, b2)
            b_cvi = prod(test[:method], b1, b2)
            @test isapprox(mean(b_analytical), mean(b_cvi), atol = test[:tol])

            beta_1 = Beta(abs(randn(rng)) + 1, abs(randn(rng)) + 1)
            beta_2 = Beta(abs(randn(rng)) + 1, abs(randn(rng)) + 1)

            beta_analytical = prod(GenericProd(), beta_1, beta_2)
            beta_cvi = prod(test[:method], beta_1, beta_2)
            @test all(isapprox.(mean_var(beta_cvi), mean_var(beta_analytical), atol = test[:tol]))
        end
    end

    @testset "Categorical x Categorical" begin
        rng = StableRNG(42)

        method = CVI(StableRNG(42), 1, 1000, Optimisers.Descent(0.007), ForwardDiffGrad(), 10, Val(true), true)

        c1 = Categorical(softmax(rand(rng, 3)))
        c2 = Categorical(softmax(rand(rng, 3)))

        c_analytical = prod(GenericProd(), c1, c2)
        c_cvi = prod(method, c1, c2)

        @test probvec(c_analytical) ≈ probvec(c_cvi) atol = 1e-1
    end

    @testset "cvi `prod` tests (n_gradpoints = 60)" begin
        rng = StableRNG(42)

        tests = (
            (method = CVI(StableRNG(42), 1, 600, Optimisers.Descent(0.01), ForwardDiffGrad(), 60, Val(true), true), tol = 2e-1),
            (method = CVI(StableRNG(42), 1, 600, Optimisers.Descent(0.01), ZygoteGrad(), 60, Val(true), true), tol = 2e-1)
        )

        # Check several prods against their analytical solutions
        for test in tests, i in 1:5

            # Univariate `Gamma`
            g1 = GammaShapeRate(rand(rng) + 1, rand(rng) + 1)
            g2 = GammaShapeRate(rand(rng) + 1, rand(rng) + 1)

            g_analytical = prod(GenericProd(), g1, g2)
            g_cvi1 = prod(test[:method], g1, g2)
            g_cvi2 = prod(test[:method], ContinuousUnivariateLogPdf((x) -> logpdf(g1, x)), g2)

            @test g_cvi1 ≈ g_analytical atol = test[:tol]
            @test g_cvi2 ≈ g_analytical atol = test[:tol]
        end
    end

    @testset "Normal x Normal (Log-likelihood preconditioner prod)" begin
        seed = 123
        rng = StableRNG(seed)
        optimizer = Optimisers.Descent(0.01)
        meta = CVI(rng, 1, 1000, optimizer, ForwardDiffGrad(), 1, Val(false), true)

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
            optimizer = Optimisers.Descent(0.001)
            meta = CVI(rng, 1, 5000, optimizer, ForwardDiffGrad(), 1, Val(false), true)

            for i in 1:3
                m_out, m_in = NormalMeanVariance(i, 1), NormalMeanVariance(0, 1)
                λ = Base.@invoke prod(meta::CVI, (ContinuousUnivariateLogPdf((z) -> logpdf(m_out, z)))::AbstractContinuousGenericLogPdf, m_in::Any)
                @test isapprox(convert(Distribution, λ), NormalWeightedMeanPrecision(i, 2), atol = 0.5)
            end
        end
    end

    @static if VERSION ≥ v"1.7" # Base.@invoke is available only in Julia >= 1.7
        @testset "MvNormal x MvNormal 1D (Fisher preconditioner prod)" begin
            seed = 123
            rng = StableRNG(seed)
            optimizer = Optimisers.Descent(0.001)
            meta = CVI(rng, 1, 5000, optimizer, ForwardDiffGrad(), 1, Val(false), true)

            for i in 1:3
                m_out, m_in = MvNormalMeanCovariance([i], [1]), MvNormalMeanCovariance([0], [1])
                λ = Base.@invoke prod(
                    meta::CVI, (ContinuousMultivariateLogPdf(ReactiveMP.UnspecifiedDomain(), (x) -> logpdf(m_out, x)))::AbstractContinuousGenericLogPdf, m_in::Any
                )
                @test isapprox(convert(Distribution, λ), MvNormalWeightedMeanPrecision([i], [2]), atol = 0.5)
            end
        end

        @testset "MvNormal x MvNormal 2D (Fisher preconditioner prod)" begin
            seed = 123
            rng = StableRNG(seed)
            optimizer = Optimisers.Descent(0.001)
            meta = CVI(rng, 1, 5000, optimizer, ForwardDiffGrad(), 10, Val(false), true)
            for i in 1:3
                m_out, m_in = MvGaussianMeanCovariance(fill(i, 2)), MvGaussianMeanCovariance(zeros(2))
                g_cvi1 = Base.@invoke prod(meta::CVI, (ContinuousMultivariateLogPdf(2, (x) -> logpdf(m_out, x)))::AbstractContinuousGenericLogPdf, m_in::Any)
                g_analytical = MvNormalWeightedMeanPrecision(fill(i, 2), diageye(2) * 2)
                @test all(isapprox.(mean(g_analytical), mean(g_cvi1), atol = 0.2))
            end
        end
    end
end
end
