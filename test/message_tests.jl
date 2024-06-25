@testitem "Message" begin
    using Random, ReactiveMP, BayesBase, Distributions, ExponentialFamily

    import InteractiveUtils: methodswith
    import Base: methods
    import Base.Iterators: repeated, product
    import BayesBase: xtlog, mirrorlog
    import ReactiveMP: getaddons, multiply_messages, as_message
    import SpecialFunctions: loggamma

    @testset "Default methods" begin
        for clamped in (true, false), initial in (true, false), addons in (1, 2), data in (1, 1.0, Normal(0, 1), Gamma(1, 1), PointMass(1))
            msg = Message(data, clamped, initial, addons)
            @test getdata(msg) === data
            @test is_clamped(msg) === clamped
            @test is_initial(msg) === initial
            @test as_message(msg) === msg
            @test getaddons(msg) === addons
            @test occursin("Message", repr(msg))
            @test occursin(repr(data), repr(msg))
            @test occursin(repr(addons), repr(msg))
        end

        dist1 = NormalMeanVariance(0.0, 1.0)
        dist2 = MvNormalMeanCovariance([0.0, 1.0], [1.0 0.0; 0.0 1.0])

        for clamped1 in (true, false), clamped2 in (true, false), initial1 in (true, false), initial2 in (true, false)
            msg1 = Message(dist1, clamped1, initial1, nothing)
            msg2 = Message(dist2, clamped2, initial2, nothing)

            @test getdata((msg1, msg2)) === (dist1, dist2)
            @test is_clamped((msg1, msg2)) === all([clamped1, clamped2])
            @test is_initial((msg1, msg2)) === all([initial1, initial2])
        end
    end

    @testset "multiply_messages" begin
        × = (x, y) -> multiply_messages(GenericProd(), x, y)

        dist1 = NormalMeanVariance(randn(), rand())
        dist2 = NormalMeanVariance(randn(), rand())

        @test getdata(Message(dist1, false, false, nothing) × Message(dist2, false, false, nothing)) == prod(GenericProd(), dist1, dist2)
        @test getdata(Message(dist2, false, false, nothing) × Message(dist1, false, false, nothing)) == prod(GenericProd(), dist2, dist1)

        for (left_is_initial, right_is_initial) in product(repeated([true, false], 2)...)
            @test is_clamped(Message(dist1, true, left_is_initial, nothing) × Message(dist2, false, right_is_initial, nothing)) == false
            @test is_clamped(Message(dist1, false, left_is_initial, nothing) × Message(dist2, true, right_is_initial, nothing)) == false
            @test is_clamped(Message(dist1, true, left_is_initial, nothing) × Message(dist2, true, right_is_initial, nothing)) == true
            @test is_clamped(Message(dist2, true, left_is_initial, nothing) × Message(dist1, false, right_is_initial, nothing)) == false
            @test is_clamped(Message(dist2, false, left_is_initial, nothing) × Message(dist1, true, right_is_initial, nothing)) == false
            @test is_clamped(Message(dist2, true, left_is_initial, nothing) × Message(dist1, true, right_is_initial, nothing)) == true
        end

        for (left_is_clamped, right_is_clamped) in product(repeated([true, false], 2)...)
            @test is_initial(Message(dist1, left_is_clamped, true, nothing) × Message(dist2, right_is_clamped, true, nothing)) == !(left_is_clamped && right_is_clamped)
            @test is_initial(Message(dist2, left_is_clamped, true, nothing) × Message(dist1, right_is_clamped, true, nothing)) == !(left_is_clamped && right_is_clamped)
            @test is_initial(Message(dist1, left_is_clamped, false, nothing) × Message(dist2, right_is_clamped, false, nothing)) == false
            @test is_initial(Message(dist2, left_is_clamped, false, nothing) × Message(dist1, right_is_clamped, false, nothing)) == false
        end

        @test is_initial(Message(dist1, true, true, nothing) × Message(dist2, true, true, nothing)) == false
        @test is_initial(Message(dist1, true, true, nothing) × Message(dist2, true, false, nothing)) == false
        @test is_initial(Message(dist1, true, false, nothing) × Message(dist2, true, true, nothing)) == false
        @test is_initial(Message(dist1, false, true, nothing) × Message(dist2, true, false, nothing)) == true
        @test is_initial(Message(dist1, true, false, nothing) × Message(dist2, false, true, nothing)) == true
        @test is_initial(Message(dist2, true, true, nothing) × Message(dist1, true, true, nothing)) == false
        @test is_initial(Message(dist2, true, true, nothing) × Message(dist1, true, false, nothing)) == false
        @test is_initial(Message(dist2, true, false, nothing) × Message(dist1, true, true, nothing)) == false
        @test is_initial(Message(dist2, false, true, nothing) × Message(dist1, true, false, nothing)) == true
        @test is_initial(Message(dist2, true, false, nothing) × Message(dist1, false, true, nothing)) == true
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
            message = Message(distribution, false, false, nothing)
            # Here we check that a specialised method for a particular type T exist
            ms = methods(method, (T,))
            if !isempty(ms) && all(m -> m ∈ distribution_methods, ms)
                @test method(message) == method(distribution)
            end
        end

        fn_mean_functions = (inv, log, xtlog, mirrorlog, loggamma)

        for distribution in distributions, fn_mean in fn_mean_functions
            F       = typeof(fn_mean)
            T       = typeof(distribution)
            message = Message(distribution, false, false, nothing)
            # Here we check that a specialised method for a particular type T exist
            ms = methods(mean, (F, T), ReactiveMP)
            if !isempty(ms)
                @test mean(fn_mean, message) == mean(fn_mean, distribution)
            end
        end

        _getpoint(rng, distribution) = _getpoint(rng, variate_form(typeof(distribution)), distribution)
        _getpoint(rng, ::Type{<:Univariate}, distribution) = 10rand(rng)
        _getpoint(rng, ::Type{<:Multivariate}, distribution) = 10 .* rand(rng, 2)

        distributions2 = [Gamma(10.0, 2.0), NormalMeanVariance(-10.0, 1.0), MvNormalMeanPrecision([2.0, -1.0], [7.0 -1.0; -1.0 3.0]), Bernoulli(0.5), Categorical([0.8, 0.2])]

        methods_to_test2 = [Distributions.pdf, Distributions.logpdf]

        rng = MersenneTwister(1234)

        for distribution in distributions2, method in methods_to_test2
            message = Message(distribution, false, false, nothing)

            for _ in 1:3
                point = _getpoint(rng, distribution)
                @test method(message, point) === method(distribution, point)
            end
        end
    end
end

@testitem "Deffered message" begin
    using Rocket
    import ReactiveMP: DefferedMessage, as_message, getdata

    for a in rand(3), b in rand(3)
        messages_stream = RecentSubject(Float64)
        marginals_stream = RecentSubject(Float64)

        dmessage = DefferedMessage(messages_stream, marginals_stream, (a, b) -> Message(a + b, false, false, nothing))

        # The data cannot be computed since no values were provided yet
        @test_throws MethodError as_message(dmessage)
        @test occursin("DeferredMessage([ use `as_message` to compute the message ])", repr(dmessage))

        next!(messages_stream, a)
        next!(marginals_stream, b)

        # The data should be computed after the values are available
        @test getdata(as_message(dmessage)) === a + b
        @test occursin("DeferredMessage($(a + b))", repr(dmessage))

        next!(messages_stream, a + 1)
        next!(marginals_stream, b + 1)

        # The data should not be changed after the first materialization
        @test getdata(as_message(dmessage)) === a + b
        @test occursin("DeferredMessage($(a + b))", repr(dmessage))
    end
end

@testitem "MessageMapping should call `rulefallback` is no rule is available" begin
    import ReactiveMP: MessageMapping, getdata

    struct SomeArbitraryNode end

    @node SomeArbitraryNode Stochastic [out, in]

    struct NonexistingDistribution end

    meta = "meta"
    addons = ()

    mapping_no_rule_fallback = MessageMapping(SomeArbitraryNode, Val(:out), Marginalisation(), Val((:in,)), nothing, meta, addons, SomeArbitraryNode(), nothing)

    messages  = (Message(NonexistingDistribution(), false, false, nothing),)
    marginals = nothing

    @test_throws ReactiveMP.RuleMethodError mapping_no_rule_fallback(messages, marginals)

    rulefallback = (args...) -> (args, nothing)

    mapping_with_fallback = MessageMapping(SomeArbitraryNode, Val(:out), Marginalisation(), Val((:in,)), nothing, meta, addons, SomeArbitraryNode(), rulefallback)

    @test getdata(mapping_with_fallback(messages, marginals)) ==
        (SomeArbitraryNode, Val(:out), Marginalisation(), Val((:in,)), messages, nothing, marginals, meta, addons, SomeArbitraryNode())
end