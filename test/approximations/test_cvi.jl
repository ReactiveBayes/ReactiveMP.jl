module ReactiveMPRenderCVITest

using Test, ReactiveMP, Random, StableRNGs, BayesBase, Distributions, ExponentialFamily, Zygote, Optimisers, DiffResults, LinearAlgebra

import BayesBase: AbstractContinuousGenericLogPdf
import StatsFuns: logistic, softmax
import SpecialFunctions: polygamma

struct NoopOptimiser end

function ReactiveMP.cvi_setup(::NoopOptimiser, λ)
    return (NoopOptimiser(), nothing)
end

function ReactiveMP.cvi_update!(state::Tuple{NoopOptimiser, Nothing}, new_λ, λ, ∇)
    return state, vec(λ)
end

mutable struct CountingOptimizer
    num_its::Int
end

function ReactiveMP.cvi_setup(opt::CountingOptimizer, λ)
    return (opt, nothing)
end

function ReactiveMP.cvi_update!(state::Tuple{CountingOptimizer, Nothing}, new_λ, λ, ∇)
    opt = state[1]
    opt.num_its += 1
    return (state, vec(λ))
end

function gammafisher(dist::GammaShapeRate)
    return [polygamma(1, shape(dist)) -1/rate(dist); -1/rate(dist) shape(dist)/rate(dist)^2]
end

@testset "cvi:prod" begin

    # These are tested in the `ExponentialFamily`, test a simple case just to be sure
    @testset "Simple check for the existence of the `fisherinformation`" begin
        @testset "GammaShapeRate" begin
            for i in 1:10, j in 1:10
                distribution = GammaShapeRate(i, j)
                expform = convert(ExponentialFamilyDistribution, distribution)
                F = fisherinformation(expform)
                @test [1 0; 0 -1]' * F * [1 0; 0 -1] ≈ gammafisher(distribution)
            end
        end
    end

    @testset "Different gradient strategies" begin
        for strategy in (ForwardDiffGrad(), ZygoteGrad())
            outputv = [0.0]
            outputm = [0.0;;]

            for i in 1:100
                @test ReactiveMP.compute_gradient!(strategy, outputv, (x) -> sum(x)^2, Float64[i]) ≈ [2 * i]
                @test ReactiveMP.compute_hessian!(strategy, outputm, (x) -> sum(x)^2, Float64[i]) ≈ [2;;]
            end
        end
    end

    @testset "Checking that the procedure runs for different parameters (with a noop-optimiser)" begin
        for strategy in (ForwardDiffGrad(), ZygoteGrad()), force_proper in (Val(true), Val(false)), warn in (true, false), n_iters in 1:3, n_gradpoints in 1:3
            for (left, right) in ((NormalMeanVariance(0, 1), NormalMeanVariance(0, 1)), (GammaShapeRate(2, 2), GammaShapeRate(2, 2)), (Bernoulli(0.5), Bernoulli(0.5)))
                method = CVI(1, n_iters, NoopOptimiser(), strategy, n_gradpoints, force_proper, warn)
                # `warn = true` reports that the iterations did not convertge, 
                # but that is normal for the no-op optimiser
                Base.with_logger(Base.NullLogger()) do
                    result = prod(method, left, right)
                    # The optimiser is no-op, the result should be equal to the right
                    @test all(mean_var(result) .== mean_var(right))
                end
            end
        end
    end

    @testset "Checking that the procedure runs for different parameters (with a counting-optimiser)" begin
        for strategy in (ForwardDiffGrad(), ZygoteGrad()), force_proper in (Val(true), Val(false)), warn in (true, false), n_iters in 1:3, n_gradpoints in 1:3
            for (left, right) in ((NormalMeanVariance(0, 1), NormalMeanVariance(0, 1)), (GammaShapeRate(2, 2), GammaShapeRate(2, 2)), (Bernoulli(0.5), Bernoulli(0.5)))
                opt = CountingOptimizer(0)
                method = CVI(1, n_iters, opt, strategy, n_gradpoints, force_proper, warn)
                # `warn = true` reports that the iterations did not convertge, 
                # but that is normal for the no-op optimiser
                Base.with_logger(Base.NullLogger()) do
                    result = prod(method, left, right)
                    # The optimiser is no-op, the result should be equal to the right
                    @test all(mean_var(result) .== mean_var(right))
                    # Checking the number of iterations
                    @test opt.num_its === n_iters
                end
            end
        end
    end

    @testset "Checking that the procedure runs for different parameters (with a lambda based counting-optimiser)" begin
        for strategy in (ForwardDiffGrad(), ZygoteGrad()), force_proper in (Val(true), Val(false)), warn in (true, false), n_iters in 1:3, n_gradpoints in 1:3
            for (left, right) in ((NormalMeanVariance(0, 1), NormalMeanVariance(0, 1)), (GammaShapeRate(2, 2), GammaShapeRate(2, 2)), (Bernoulli(0.5), Bernoulli(0.5)))
                counting = 0
                callback = (new_λ, λ, _) -> begin
                    counting += 1
                    return λ
                end

                method = CVI(1, n_iters, callback, strategy, n_gradpoints, force_proper, warn)
                # `warn = true` reports that the iterations did not convertge, 
                # but that is normal for the no-op optimiser
                Base.with_logger(Base.NullLogger()) do
                    result = prod(method, left, right)
                    # The optimiser is no-op, the result should be equal to the right
                    @test all(mean_var(result) .== mean_var(right))
                    # Checking the number of iterations
                    @test counting === n_iters
                end
            end
        end
    end

    @testset "cvi `prod` tests" begin
        rng = StableRNG(42)

        for i in 1:5
            # Here candidates is a collection of tests 
            # `left` - left message 
            # `right` - right message 
            # `grads` - a collection of gradient strategies (by default only `ForwardDiffGrad()`, `Zygote` is too slow)
            # `optimisers` - a collection of optimisers (by default `Optimisers.Descent(0.007)`)
            # `n_iters` - a collection of number of iterations (by default `1000`)
            # `n_gradpoints` - a collection of number of gradient points (by default `50`)
            # `tol` - an absolute tolerance (by default `1e-2`)
            candidates = (
                (left = Bernoulli(rand(rng)), right = Bernoulli(rand(rng)), tol = 9e-3),
                (left = Beta(abs(randn(rng)) + 1, abs(randn(rng)) + 1), right = Beta(abs(randn(rng)) + 1, abs(randn(rng)) + 1), tol = 1e-2),
                (left = GammaShapeRate(rand(rng) + 1, rand(rng) + 1), right = GammaShapeRate(rand(rng) + 1, rand(rng) + 1), tol = 8e-2),
                (left = GammaShapeScale(rand(rng) + 1, rand(rng) + 1), right = GammaShapeScale(rand(rng) + 1, rand(rng) + 1), tol = 1e-1)
            )
            # This list is not exhaustive in any way
            for candidate in candidates
                grads = get(candidate, :grads, (ForwardDiffGrad(),))
                optimisers = get(candidate, :optimisers, (Optimisers.Descent(0.007),))
                n_iters = get(candidate, :n_iters, (1000,))
                n_gradpoints = get(candidate, :n_gradpoints, (50,))
                tol = get(candidate, :tol, 1e-2)

                left = candidate[:left]
                right = candidate[:right]
                closed = prod(GenericProd(), left, right)

                for grad in grads, optimiser in optimisers, n in n_iters, k in n_gradpoints
                    method = CVI(StableRNG(42), 1, n, optimiser, grad, k, Val(true), true)
                    numerical_1 = prod(method, left, right)
                    numerical_2 = prod(method, ContinuousUnivariateLogPdf((x) -> logpdf(left, x)), right)
                    for numerical in (numerical_1, numerical_2)
                        @test all(isapprox(mean(numerical), mean(closed), atol = tol))
                        # For the univariate case additionaly check the variance
                        if variate_form(typeof(left)) === Univariate && variate_form(typeof(right)) === Univariate
                            @test all(isapprox(var(numerical), var(closed), atol = tol))
                        end
                        # For the univariate case additionaly check the covariance
                        if variate_form(typeof(left)) === Multivariate && variate_form(typeof(right)) === Multivariate
                            @test all(isapprox(cov(numerical), cov(closed), atol = tol))
                        end
                    end
                end
            end
        end

        # Check several prods against their analytical solutions
        # for test in tests, i in 1:5

        # Univariate `Normal`
        # n1 = NormalMeanVariance(10 * randn(rng), 10 * rand(rng))
        # n2 = NormalMeanVariance(10 * randn(rng), 10 * rand(rng))

        # n_analytical = prod(GenericProd(), n1, n2)

        # @test prod(test[:method], ContinuousUnivariateLogPdf((x) -> logpdf(n1, x)), n2) ≈ n_analytical atol = test[:tol]
        # @test prod(test[:method], n1, n2) ≈ n_analytical atol = test[:tol]

        # Univariate `Gamma`
        # g1 = GammaShapeRate(rand(rng) + 1, rand(rng) + 1)
        # g2 = GammaShapeRate(rand(rng) + 1, rand(rng) + 1)

        # g_analytical = prod(GenericProd(), g1, g2)
        # g_cvi1 = prod(test[:method], g1, g2)
        # g_cvi2 = prod(test[:method], ContinuousUnivariateLogPdf((x) -> logpdf(g1, x)), g2)

        # @test all(isapprox.(mean_var(g_analytical), mean_var(g_cvi1), atol = test[:tol]))
        # @test all(isapprox.(mean_var(g_analytical), mean_var(g_cvi2), atol = test[:tol]))

        # Multivariate `Normal`
        # if !(test[:method].grad isa ZygoteGrad) # `Zygote` does not support mutations
        #     for d in (2, 3)
        #         mn1 = MvNormalMeanCovariance(10 * randn(rng, d), 10 * rand(rng, d))
        #         mn2 = MvNormalMeanCovariance(10 * randn(rng, d), 10 * rand(rng, d))

        #         mn_analytical = prod(GenericProd(), mn1, mn2)

        #         @test prod(test[:method], mn1, mn2) ≈ mn_analytical atol = test[:tol]
        #         @test prod(test[:method], ContinuousMultivariateLogPdf(d, (x) -> logpdf(mn1, x)), mn2) ≈ mn_analytical atol = test[:tol]
        #     end
        # end

        # b1 = Bernoulli(logistic(randn(rng)))
        # b2 = Bernoulli(logistic(randn(rng)))
        # b_analytical = prod(GenericProd(), b1, b2)
        # b_cvi = prod(test[:method], b1, b2)
        # @test isapprox(mean(b_analytical), mean(b_cvi), atol = test[:tol])

        # beta_1 = Beta(abs(randn(rng)) + 1, abs(randn(rng)) + 1)
        # beta_2 = Beta(abs(randn(rng)) + 1, abs(randn(rng)) + 1)

        # beta_analytical = prod(GenericProd(), beta_1, beta_2)
        # beta_cvi = prod(test[:method], beta_1, beta_2)
        # @test all(isapprox.(mean_var(beta_cvi), mean_var(beta_analytical), atol = test[:tol]))
        #     end
    end

    # @testset "Categorical x Categorical" begin
    #     rng = StableRNG(42)

    #     method = CVI(StableRNG(42), 1, 1000, Optimisers.Descent(0.007), ForwardDiffGrad(), 10, Val(true), true)

    #     c1 = Categorical(softmax(rand(rng, 3)))
    #     c2 = Categorical(softmax(rand(rng, 3)))

    #     c_analytical = prod(GenericProd(), c1, c2)
    #     c_cvi = prod(method, c1, c2)

    #     @test probvec(c_analytical) ≈ probvec(c_cvi) atol = 1e-1
    # end

    # @testset "Normal x Normal (Log-likelihood preconditioner prod)" begin
    #     seed = 123
    #     rng = StableRNG(seed)
    #     optimizer = Optimisers.Descent(0.01)
    #     meta = CVI(rng, 1, 1000, optimizer, ForwardDiffGrad(), 1, Val(false), true)

    #     for i in 1:10
    #         m_out, m_in = NormalMeanVariance(i, 1), NormalMeanVariance(0, 1)
    #         λ = prod(meta, ContinuousUnivariateLogPdf((z) -> logpdf(m_out, z)), m_in)
    #         @test isapprox(convert(Distribution, λ), NormalWeightedMeanPrecision(i, 2), atol = 0.1)
    #     end
    # end

    # Extra tests for non-generic Gaussians
    @static if VERSION ≥ v"1.7" # Base.@invoke is available only in Julia >= 1.7
        @testset "Normal x Normal (Fisher preconditioner prod)" begin
            seed = 123
            rng = StableRNG(seed)
            optimizer = Optimisers.Descent(0.001)
            meta = CVI(rng, 1, 5000, optimizer, ForwardDiffGrad(), 10, Val(false), true)

            for i in 1:3, j in 1:3
                m_out, m_in = NormalMeanVariance(i, 1 + j), NormalMeanVariance(-i, 1 + j^2)
                closed = prod(ClosedProd(), m_out, m_in)
                numerical = Base.@invoke prod(meta::CVI, m_out::Any, m_in::Any)
                @test isapprox(mean(numerical), mean(closed), atol = 7e-2)
                @test isapprox(var(numerical), var(closed), atol = 7e-1)
            end
        end
    end

    @static if VERSION ≥ v"1.7" # Base.@invoke is available only in Julia >= 1.7
        @testset "MvNormal x MvNormal 1D (Fisher preconditioner prod)" begin
            seed = 123
            rng = StableRNG(seed)
            optimizer = Optimisers.Descent(0.001)
            meta = CVI(rng, 1, 5000, optimizer, ForwardDiffGrad(), 10, Val(false), true)

            for i in 1:3, j in 1:3
                m_out, m_in = MvNormalMeanCovariance([i], [1 + j]), MvNormalMeanCovariance([-i], [1 + j^2])
                closed = prod(ClosedProd(), m_out, m_in)
                numerical = Base.@invoke prod(meta::CVI, m_out::Any, m_in::Any)
                @test isapprox(mean(numerical), mean(closed), atol = 7e-2)
                @test isapprox(cov(numerical), cov(closed), atol = 7e-1)
            end
        end

        @testset "MvNormal x MvNormal 2D (Fisher preconditioner prod)" begin
            seed = 123
            rng = StableRNG(seed)
            optimizer = Optimisers.Descent(0.001)
            meta = CVI(rng, 1, 5000, optimizer, ForwardDiffGrad(), 10, Val(false), false)
            for i in 1:3, j in 1:3
                m_out, m_in = MvGaussianMeanCovariance(fill(i, 2), Matrix(Diagonal(fill(1 + j, 2)))), MvGaussianMeanCovariance(fill(-i, 2), Matrix(Diagonal(fill(1 + j^2, 2))))
                closed = prod(ClosedProd(), m_out, m_in)
                numerical = Base.@invoke prod(meta::CVI, m_out::Any, m_in::Any)
                # These tests are broken because of the unconstrained optimization
                # When fixed, set `warn = true` in the `CVI` constructor
                @test_broken isapprox(mean(numerical), mean(closed), atol = 7e-2)
                @test_broken isapprox(cov(numerical), cov(closed), atol = 7e-1)
            end
        end
    end
end
end
