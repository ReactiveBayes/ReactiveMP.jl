
@testitem "cvi:prod" begin
    using ReactiveMP, Random, StableRNGs, BayesBase, Distributions, ExponentialFamily, Optimisers, DiffResults, LinearAlgebra

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

    @testset "Checking that the procedure runs for different parameters (with a noop-optimiser)" begin
        for strategy in (ForwardDiffGrad(), ForwardDiffGrad(1)), force_proper in (Val(true), Val(false)), warn in (true, false), n_iters in 1:3, n_gradpoints in 1:3
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
        for strategy in (ForwardDiffGrad(), ForwardDiffGrad(1)), force_proper in (Val(true), Val(false)), warn in (true, false), n_iters in 1:3, n_gradpoints in 1:3
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
        for strategy in (ForwardDiffGrad(), ForwardDiffGrad(1)), force_proper in (Val(true), Val(false)), warn in (true, false), n_iters in 1:3, n_gradpoints in 1:3
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

    @testset "Simple products compared to their analytical solutions" begin
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
                # (left = Categorical(softmax(rand(rng, 3))), Categorical(softmax(rand(rng, 3)))) # Categorical is broken, needs fix!
            )
            # This list is not exhaustive in any way, e.g. Gaussians are tested below
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
                    # TODO: This would require `ContinuousMultivariateLogPdf(UnspecifiedDomain(), (x) -> logpdf(left, x))`
                    # for multivariate inputs
                    numerical_2 = prod(method, ContinuousUnivariateLogPdf((x) -> logpdf(left, x)), right)
                    for numerical in (numerical_1, numerical_2)
                        @test all(isapprox(mean(numerical), mean(closed), atol = tol))
                        # For the univariate case additionaly check the variance
                        if variate_form(typeof(left)) === Univariate && variate_form(typeof(right)) === Univariate
                            @test all(isapprox(var(numerical), var(closed), atol = tol))
                        end
                        # For the multivariate case additionaly check the covariance
                        if variate_form(typeof(left)) === Multivariate && variate_form(typeof(right)) === Multivariate
                            @test all(isapprox(cov(numerical), cov(closed), atol = tol))
                        end
                    end
                end
            end
        end
    end

    @testset "Normal x Normal (Log-likelihood preconditioner prod)" begin
        seed = 123
        rng = StableRNG(seed)
        optimizer = Optimisers.Descent(0.007)
        meta = CVI(rng, 1, 1000, optimizer, ForwardDiffGrad(), 20, Val(false), true)

        for i in 1:5, j in 1:5
            left = NormalMeanVariance(randn(rng), 1 + j + rand(rng))
            right = NormalMeanVariance(-i, 1 + j^2)
            closed = prod(ClosedProd(), left, right)
            numerical_1 = prod(meta, left, right)
            numerical_2 = prod(meta, ContinuousUnivariateLogPdf((x) -> logpdf(left, x)), right)
            for numerical in (numerical_1, numerical_2)
                @test isapprox(mean(numerical), mean(closed), atol = 5e-2)
                @test isapprox(var(numerical), var(closed), atol = 5e-2)
            end
        end
    end

    @testset "MvNormal x MvNormal (Log-likelihood preconditioner prod)" begin
        seed = 123
        rng = StableRNG(seed)
        optimizer = Optimisers.Descent(0.007)
        meta = CVI(rng, 1, 1000, optimizer, ForwardDiffGrad(), 20, Val(false), true)

        for n in 1:10, i in 1:5, j in 1:5
            L = LowerTriangular(rand(rng, n, n)) + n * I
            Σ = L * L'
            left = MvNormalMeanPrecision(rand(rng, n), Σ)
            right = MvNormalMeanCovariance(fill(-i, n), fill(1 + j^2, n))
            closed = prod(ClosedProd(), left, right)
            numerical_1 = prod(meta, left, right)
            numerical_2 = prod(meta, ContinuousMultivariateLogPdf(UnspecifiedDomain(), (x) -> logpdf(left, x)), right)
            for numerical in (numerical_1, numerical_2)
                @test isapprox(mean(numerical), mean(closed), atol = n * 5e-2, rtol = 5e-2)
                @test isapprox(cov(numerical), cov(closed), atol = n * 5e-2, rtol = 5e-2)
            end
        end
    end
end
