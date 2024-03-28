@testitem "Marginal" begin
    using Random, ReactiveMP, BayesBase, Distributions, ExponentialFamily

    import InteractiveUtils: methodswith
    import Base: methods
    import Base.Iterators: repeated, product
    import BayesBase: xtlog, mirrorlog
    import ReactiveMP: getaddons, as_marginal
    import SpecialFunctions: loggamma

    @testset "Default methods" begin
        for clamped in (true, false), initial in (true, false), addons in (1, 2), data in (1, 1.0, Normal(0, 1), Gamma(1, 1), PointMass(1))
            marginal = Marginal(data, clamped, initial, addons)
            @test getdata(marginal) === data
            @test is_clamped(marginal) === clamped
            @test is_initial(marginal) === initial
            @test as_marginal(marginal) === marginal
            @test getaddons(marginal) === addons
            @test occursin("Marginal", repr(marginal))
            @test occursin(repr(data), repr(marginal))
            @test occursin(repr(addons), repr(marginal))
        end

        dist1 = NormalMeanVariance(0.0, 1.0)
        dist2 = MvNormalMeanCovariance([0.0, 1.0], [1.0 0.0; 0.0 1.0])

        for clamped1 in (true, false), clamped2 in (true, false), initial1 in (true, false), initial2 in (true, false)
            msg1 = Marginal(dist1, clamped1, initial1, nothing)
            msg2 = Marginal(dist2, clamped2, initial2, nothing)

            @test getdata((msg1, msg2)) === (dist1, dist2)
            @test is_clamped((msg1, msg2)) === all([clamped1, clamped2])
            @test is_initial((msg1, msg2)) === all([initial1, initial2])
        end
    end

    @testset "Statistics" begin
        distributions = [
            PointMass(0.5),
            Gamma(10.0, 2.0),
            NormalMeanVariance(-10.0, 10.0),
            Wishart(4.0, [2.0 -0.5; -0.5 1.0]),
            MvNormalMeanPrecision([2.0, -1.0], [7.0 -1.0; -1.0 3.0]),
            Bernoulli(0.5),
            Categorical([0.8, 0.2])
        ]

        # Here we get all methods defined for a particular type of a distribution
        dists_methods = map(d -> methodswith(eval(nameof(typeof(d)))), distributions)

        methods_to_test = [
            BayesBase.mean,
            BayesBase.median,
            BayesBase.mode,
            BayesBase.shape,
            BayesBase.scale,
            BayesBase.rate,
            BayesBase.var,
            BayesBase.std,
            BayesBase.cov,
            BayesBase.invcov,
            BayesBase.logdetcov,
            BayesBase.entropy,
            BayesBase.params,
            BayesBase.mean_cov,
            BayesBase.mean_var,
            BayesBase.mean_invcov,
            BayesBase.mean_precision,
            BayesBase.weightedmean_cov,
            BayesBase.weightedmean_var,
            BayesBase.weightedmean_invcov,
            BayesBase.weightedmean_precision,
            BayesBase.probvec,
            BayesBase.weightedmean,
            Base.precision,
            Base.length,
            Base.ndims,
            Base.size,
            Base.eltype
        ]

        for (distribution, distribution_methods) in zip(distributions, dists_methods), method in methods_to_test
            T       = typeof(distribution)
            marginal = Marginal(distribution, false, false, nothing)
            # Here we check that a specialised method for a particular type T exist
            ms = methods(method, (T,))
            if !isempty(ms) && all(m -> m ∈ distribution_methods, ms)
                @test method(marginal) == method(distribution)
            end
        end

        fn_mean_functions = (inv, log, xtlog, mirrorlog, loggamma)

        for distribution in distributions, fn_mean in fn_mean_functions
            F       = typeof(fn_mean)
            T       = typeof(distribution)
            marginal = Marginal(distribution, false, false, nothing)
            # Here we check that a specialised method for a particular type T exist
            ms = methods(mean, (F, T), ReactiveMP)
            if !isempty(ms)
                @test mean(fn_mean, marginal) == mean(fn_mean, distribution)
            end
        end

        _getpoint(rng, distribution) = _getpoint(rng, variate_form(typeof(distribution)), distribution)
        _getpoint(rng, ::Type{<:Univariate}, distribution) = 10rand(rng)
        _getpoint(rng, ::Type{<:Multivariate}, distribution) = 10 .* rand(rng, 2)

        distributions2 = [Gamma(10.0, 2.0), NormalMeanVariance(-10.0, 1.0), MvNormalMeanPrecision([2.0, -1.0], [7.0 -1.0; -1.0 3.0]), Bernoulli(0.5), Categorical([0.8, 0.2])]

        methods_to_test2 = [Distributions.pdf, Distributions.logpdf]

        rng = MersenneTwister(1234)

        for distribution in distributions2, method in methods_to_test2
            marginal = Marginal(distribution, false, false, nothing)

            for _ in 1:3
                point = _getpoint(rng, distribution)
                @test method(marginal, point) === method(distribution, point)
            end
        end
    end
end
