@testitem "Message" begin
    using Random, ReactiveMP, BayesBase, Distributions, ExponentialFamily

    import InteractiveUtils: methodswith
    import Base: methods
    import Base.Iterators: repeated, product
    import BayesBase: xtlog, mirrorlog
    import ReactiveMP:
        getannotations,
        AnnotationDict,
        compute_product_of_two_messages,
        MessageProductContext,
        as_message
    import SpecialFunctions: loggamma

    @testset "Default methods" begin
        for clamped in (true, false),
            initial in (true, false),
            data in (1, 1.0, Normal(0, 1), Gamma(1, 1), PointMass(1))

            msg = Message(data, clamped, initial)
            @test getdata(msg) === data
            @test is_clamped(msg) === clamped
            @test is_initial(msg) === initial
            @test as_message(msg) === msg
            @test getannotations(msg) isa AnnotationDict
            @test occursin("Message", repr(msg))
            @test occursin(repr(data), repr(msg))
        end

        dist1 = NormalMeanVariance(0.0, 1.0)
        dist2 = MvNormalMeanCovariance([0.0, 1.0], [1.0 0.0; 0.0 1.0])

        for clamped1 in (true, false),
            clamped2 in (true, false), initial1 in (true, false),
            initial2 in (true, false)

            msg1 = Message(dist1, clamped1, initial1)
            msg2 = Message(dist2, clamped2, initial2)

            @test getdata((msg1, msg2)) === (dist1, dist2)
            @test is_clamped((msg1, msg2)) === all([clamped1, clamped2])
            @test is_initial((msg1, msg2)) === all([initial1, initial2])
        end
    end

    @testset "compute product of two messages" begin
        _testvar = ReactiveMP.randomvar()
        × =
            (x, y) -> compute_product_of_two_messages(
                _testvar, MessageProductContext(), x, y
            )

        dist1 = NormalMeanVariance(randn(), rand())
        dist2 = NormalMeanVariance(randn(), rand())

        @test getdata(
            Message(dist1, false, false) × Message(dist2, false, false)
        ) == prod(GenericProd(), dist1, dist2)
        @test getdata(
            Message(dist2, false, false) × Message(dist1, false, false)
        ) == prod(GenericProd(), dist2, dist1)

        for (left_is_initial, right_is_initial) in
            product(repeated([true, false], 2)...)
            @test is_clamped(
                Message(dist1, true, left_is_initial) ×
                Message(dist2, false, right_is_initial),
            ) == false
            @test is_clamped(
                Message(dist1, false, left_is_initial) ×
                Message(dist2, true, right_is_initial),
            ) == false
            @test is_clamped(
                Message(dist1, true, left_is_initial) ×
                Message(dist2, true, right_is_initial),
            ) == true
            @test is_clamped(
                Message(dist2, true, left_is_initial) ×
                Message(dist1, false, right_is_initial),
            ) == false
            @test is_clamped(
                Message(dist2, false, left_is_initial) ×
                Message(dist1, true, right_is_initial),
            ) == false
            @test is_clamped(
                Message(dist2, true, left_is_initial) ×
                Message(dist1, true, right_is_initial),
            ) == true
        end

        for (left_is_clamped, right_is_clamped) in
            product(repeated([true, false], 2)...)
            @test is_initial(
                Message(dist1, left_is_clamped, true) ×
                Message(dist2, right_is_clamped, true),
            ) == !(left_is_clamped && right_is_clamped)
            @test is_initial(
                Message(dist2, left_is_clamped, true) ×
                Message(dist1, right_is_clamped, true),
            ) == !(left_is_clamped && right_is_clamped)
            @test is_initial(
                Message(dist1, left_is_clamped, false) ×
                Message(dist2, right_is_clamped, false),
            ) == false
            @test is_initial(
                Message(dist2, left_is_clamped, false) ×
                Message(dist1, right_is_clamped, false),
            ) == false
        end

        @test is_initial(
            Message(dist1, true, true) × Message(dist2, true, true)
        ) == false
        @test is_initial(
            Message(dist1, true, true) × Message(dist2, true, false)
        ) == false
        @test is_initial(
            Message(dist1, true, false) × Message(dist2, true, true)
        ) == false
        @test is_initial(
            Message(dist1, false, true) × Message(dist2, true, false)
        ) == true
        @test is_initial(
            Message(dist1, true, false) × Message(dist2, false, true)
        ) == true
        @test is_initial(
            Message(dist2, true, true) × Message(dist1, true, true)
        ) == false
        @test is_initial(
            Message(dist2, true, true) × Message(dist1, true, false)
        ) == false
        @test is_initial(
            Message(dist2, true, false) × Message(dist1, true, true)
        ) == false
        @test is_initial(
            Message(dist2, false, true) × Message(dist1, true, false)
        ) == true
        @test is_initial(
            Message(dist2, true, false) × Message(dist1, false, true)
        ) == true
    end

    @testset "Statistics" begin
        distributions = [
            PointMass(0.5),
            Gamma(10.0, 2.0),
            NormalMeanVariance(-10.0, 10.0),
            Wishart(4.0, [2.0 -0.5; -0.5 1.0]),
            MvNormalMeanPrecision([2.0, -1.0], [7.0 -1.0; -1.0 3.0]),
            Bernoulli(0.5),
            Categorical([0.8, 0.2]),
        ]

        # Here we get all methods defined for a particular type of a distribution
        dists_methods = map(
            d -> methodswith(eval(nameof(typeof(d)))), distributions
        )

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
            Base.eltype,
        ]

        for (distribution, distribution_methods) in
            zip(distributions, dists_methods),
            method in methods_to_test

            T       = typeof(distribution)
            message = Message(distribution, false, false)
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
            message = Message(distribution, false, false)
            # Here we check that a specialised method for a particular type T exist
            ms = methods(mean, (F, T), ReactiveMP)
            if !isempty(ms)
                @test mean(fn_mean, message) == mean(fn_mean, distribution)
            end
        end

        _getpoint(rng, distribution) = _getpoint(
            rng, variate_form(typeof(distribution)), distribution
        )
        _getpoint(rng, ::Type{<:Univariate}, distribution) = 10rand(rng)
        _getpoint(rng, ::Type{<:Multivariate}, distribution) =
            10 .* rand(rng, 2)

        distributions2 = [
            Gamma(10.0, 2.0),
            NormalMeanVariance(-10.0, 1.0),
            MvNormalMeanPrecision([2.0, -1.0], [7.0 -1.0; -1.0 3.0]),
            Bernoulli(0.5),
            Categorical([0.8, 0.2]),
        ]

        methods_to_test2 = [Distributions.pdf, Distributions.logpdf]

        rng = MersenneTwister(1234)

        for distribution in distributions2, method in methods_to_test2
            message = Message(distribution, false, false)

            for _ in 1:3
                point = _getpoint(rng, distribution)
                @test method(message, point) === method(distribution, point)
            end
        end
    end
end

@testitem "Deferred message" begin
    using Rocket
    import ReactiveMP: DeferredMessage, as_message, getdata

    for a in rand(3), b in rand(3)
        messages_stream = RecentSubject(Float64)
        marginals_stream = RecentSubject(Float64)

        dmessage = DeferredMessage(
            messages_stream,
            marginals_stream,
            (a, b) -> Message(a + b, false, false),
        )

        # The data cannot be computed since no values were provided yet
        @test_throws MethodError as_message(dmessage)
        @test occursin(
            "DeferredMessage([ use `as_message` to compute the message ])",
            repr(dmessage),
        )

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
    import ReactiveMP: MessageMapping, getdata, AnnotationDict

    struct SomeArbitraryNodeForRuleFallback end

    @node SomeArbitraryNodeForRuleFallback Stochastic [out, in]

    struct NonexistingDistribution end

    meta = "meta"
    annotations = nothing

    mapping_no_rule_fallback = MessageMapping(
        SomeArbitraryNodeForRuleFallback,
        Val(:out),
        Marginalisation(),
        Val((:in,)),
        nothing,
        meta,
        annotations,
        SomeArbitraryNodeForRuleFallback(),
        nothing,
        nothing,
    )

    messages  = (Message(NonexistingDistribution(), false, false),)
    marginals = nothing

    @test_throws ReactiveMP.RuleMethodError mapping_no_rule_fallback(
        messages, marginals
    )

    rulefallback = (args...) -> (args)

    mapping_with_fallback = MessageMapping(
        SomeArbitraryNodeForRuleFallback,
        Val(:out),
        Marginalisation(),
        Val((:in,)),
        nothing,
        meta,
        annotations,
        SomeArbitraryNodeForRuleFallback(),
        rulefallback,
        nothing,
    )

    @test getdata(mapping_with_fallback(messages, marginals)) == (
        SomeArbitraryNodeForRuleFallback,
        Val(:out),
        Marginalisation(),
        Val((:in,)),
        messages,
        nothing,
        marginals,
        meta,
        AnnotationDict(),
        SomeArbitraryNodeForRuleFallback(),
    )
end

@testitem "MessageMapping should call provided callbacks handler" begin
    import ReactiveMP: MessageMapping, getdata, AnnotationDict

    struct SomeArbitraryNodeCallbacksTests end

    @node SomeArbitraryNodeCallbacksTests Stochastic [out, in]

    @rule SomeArbitraryNodeCallbacksTests(:out, Marginalisation) (m_in::Int,) =
        m_in + 1

    events = []

    callbacks = (
        before_message_rule_call = (event) ->
            push!(events, (event = :before_message_rule_call, data = event)),
        after_message_rule_call = (event) ->
            push!(events, (event = :after_message_rule_call, data = event)),
    )

    mapping = MessageMapping(
        SomeArbitraryNodeCallbacksTests,
        Val(:out),
        Marginalisation(),
        Val((:in,)),
        nothing,
        nothing,
        (),
        SomeArbitraryNodeCallbacksTests(),
        nothing,
        callbacks,
    )

    messages = (Message(1, false, false),)
    marginals = nothing

    @test getdata(mapping(messages, marginals)) == 2

    @test events[1].event == :before_message_rule_call
    @test events[1].data.mapping.factornode ===
        SomeArbitraryNodeCallbacksTests()
    @test events[1].data.messages === messages
    @test events[1].data.marginals === marginals

    @test events[2].event == :after_message_rule_call
    @test events[2].data.mapping.factornode ===
        SomeArbitraryNodeCallbacksTests()
    @test events[2].data.messages === messages
    @test events[2].data.marginals === marginals
    @test events[2].data.result === 2
    @test events[2].data.annotations isa AnnotationDict
end

@testmodule MessageProductContextUtils begin
    import ReactiveMP: AbstractVariable, AbstractFormConstraint, constrain_form
    import ReactiveMP.BayesBase: prod, GenericProd, isapprox
    import ReactiveMP

    struct Normal
        mean::Float64
        var::Float64
    end

    function prod(::GenericProd, left::Normal, right::Normal)
        result_var = 1 / (1 / left.var + 1 / right.var)
        result_mean =
            result_var * (left.mean / left.var + right.mean / right.var)
        return Normal(result_mean, result_var)
    end

    function isapprox(left::Normal, right::Normal; kwargs...)
        return isapprox(left.mean, right.mean; kwargs...) &&
               isapprox(left.var, right.var; kwargs...)
    end

    struct AbstractVariableForMessageProductContextTests <: AbstractVariable end

    testvar = AbstractVariableForMessageProductContextTests()

    # A simple form constraint that adds +1 to the mean of a Normal distribution
    struct AddOneToMeanConstraint <: AbstractFormConstraint end

    constrain_form(::AddOneToMeanConstraint, dist::Normal) = Normal(
        dist.mean + 1, dist.var
    )

    # A callback handler that only records events from a specified set
    struct SaveOrderOfComputationCallbacks
        listen_to::Tuple
        events
    end

    function ReactiveMP.invoke_callback(
        handler::SaveOrderOfComputationCallbacks, event::ReactiveMP.Event{E}
    ) where {E}
        E ∈ handler.listen_to &&
            push!(handler.events, (event = E, data = event))
    end

    export Normal,
        testvar, AddOneToMeanConstraint, SaveOrderOfComputationCallbacks
end

@testitem "MessageProductContext should compute product of two messages" setup = [
    MessageProductContextUtils
] begin
    import ReactiveMP:
        Message, MessageProductContext, compute_product_of_two_messages, getdata

    context = MessageProductContext()

    msg1 = Message(Normal(0, 1), false, false)
    msg2 = Message(Normal(0, 1), false, false)

    result = @inferred(
        compute_product_of_two_messages(testvar, context, msg1, msg2)
    )

    @test result isa Message
    @test getdata(result) === Normal(0, 1 / 2)
end

@testitem "compute_message_product propagates the `is_clamped` and `is_initial` correctly" setup = [
    MessageProductContextUtils
] begin
    import ReactiveMP:
        Message,
        MessageProductContext,
        compute_product_of_two_messages,
        is_clamped,
        is_initial

    context = MessageProductContext()

    for left_is_clamped in (true, false),
        right_is_clamped in (true, false), left_is_initial in (true, false),
        right_is_initial in (true, false)

        msg1 = Message(Normal(0, 1), left_is_clamped, left_is_initial)
        msg2 = Message(Normal(0, 1), right_is_clamped, right_is_initial)

        result = @inferred(
            compute_product_of_two_messages(testvar, context, msg1, msg2)
        )

        @test result isa Message

        expected_result_is_clamped = left_is_clamped && right_is_clamped
        expected_result_is_initial =
            !expected_result_is_clamped &&
            (left_is_clamped || left_is_initial) &&
            (right_is_clamped || right_is_initial)

        @test is_clamped(result) === expected_result_is_clamped
        @test is_initial(result) === expected_result_is_initial
    end
end

@testitem "compute_message_product should support different folding strategies" setup = [
    MessageProductContextUtils
] begin
    import ReactiveMP:
        MessageProductContext,
        Message,
        compute_product_of_messages,
        compute_product_of_two_messages,
        getdata,
        AnnotationDict

    messages = [
        Message(Normal(0, 1), false, false)
        Message(Normal(0, 2), false, false)
        Message(Normal(0, 3), false, false)
    ]

    @testset "From left to right" begin
        import ReactiveMP: MessagesProductFromLeftToRight

        listen_to = (
            :before_product_of_two_messages, :after_product_of_two_messages
        )
        handler = SaveOrderOfComputationCallbacks(listen_to, [])
        context = MessageProductContext(;
            fold_strategy = MessagesProductFromLeftToRight(),
            callbacks = handler,
        )

        result = @inferred(
            compute_product_of_messages(testvar, context, messages)
        )

        @test result isa Message
        @test getdata(result) === Normal(0, 1 / (1 + 1 / 2 + 1 / 3))

        # 3 messages = 2 products, each product fires a before and after callback
        @test length(handler.events) == 4
        @test handler.events[1].event === :before_product_of_two_messages
        @test handler.events[2].event === :after_product_of_two_messages
        @test handler.events[3].event === :before_product_of_two_messages
        @test handler.events[4].event === :after_product_of_two_messages

        # First product: Normal(0,1) × Normal(0,2) — left to right order
        @test getdata(handler.events[1].data.left) == Normal(0, 1)
        @test getdata(handler.events[1].data.right) == Normal(0, 2)

        # Second product: result of first × Normal(0,3)
        @test getdata(handler.events[3].data.left) ==
            Normal(0, 1 / (1 / 1 + 1 / 2))
        @test getdata(handler.events[3].data.right) == Normal(0, 3)
    end

    @testset "From right to left" begin
        import ReactiveMP: MessagesProductFromRightToLeft

        listen_to = (
            :before_product_of_two_messages, :after_product_of_two_messages
        )
        handler = SaveOrderOfComputationCallbacks(listen_to, [])
        context = MessageProductContext(;
            fold_strategy = MessagesProductFromRightToLeft(),
            callbacks = handler,
        )

        result = @inferred(
            compute_product_of_messages(testvar, context, messages)
        )

        @test result isa Message
        @test getdata(result) === Normal(0, 1 / (1 + 1 / 2 + 1 / 3))

        # 3 messages = 2 products, each product fires a before and after callback
        @test length(handler.events) == 4
        @test handler.events[1].event === :before_product_of_two_messages
        @test handler.events[2].event === :after_product_of_two_messages
        @test handler.events[3].event === :before_product_of_two_messages
        @test handler.events[4].event === :after_product_of_two_messages

        # First product: Normal(0,2) × Normal(0,3) — right to left order
        @test getdata(handler.events[1].data.left) == Normal(0, 2)
        @test getdata(handler.events[1].data.right) == Normal(0, 3)

        # Second product: Normal(0,1) × result of first
        @test getdata(handler.events[3].data.left) == Normal(0, 1)
        @test getdata(handler.events[3].data.right) ==
            Normal(0, 1 / (1 / 2 + 1 / 3))
    end

    @testset "Custom fold strategy via Function" begin
        # Custom strategy: compute (1 × 3) × 2
        custom_fold =
            (variable, context, messages) -> begin
                first = compute_product_of_two_messages(
                    variable, context, messages[1], messages[3]
                )
                return compute_product_of_two_messages(
                    variable, context, first, messages[2]
                )
            end

        listen_to = (
            :before_product_of_two_messages, :after_product_of_two_messages
        )
        handler = SaveOrderOfComputationCallbacks(listen_to, [])
        context = MessageProductContext(;
            fold_strategy = custom_fold, callbacks = handler
        )

        result = @inferred(
            compute_product_of_messages(testvar, context, messages)
        )

        @test result isa Message
        @test getdata(result) === Normal(0, 1 / (1 + 1 / 2 + 1 / 3))

        @test length(handler.events) == 4

        # First product: Normal(0,1) × Normal(0,3)
        @test getdata(handler.events[1].data.left) == Normal(0, 1)
        @test getdata(handler.events[1].data.right) == Normal(0, 3)

        # Second product: result of first × Normal(0,2)
        @test getdata(handler.events[3].data.left) ==
            Normal(0, 1 / (1 / 1 + 1 / 3))
        @test getdata(handler.events[3].data.right) == Normal(0, 2)
    end

    @testset "Before and after callbacks receive correct arguments" begin
        import ReactiveMP: MessagesProductFromLeftToRight

        listen_to = (
            :before_product_of_two_messages, :after_product_of_two_messages
        )
        handler = SaveOrderOfComputationCallbacks(listen_to, [])
        context = MessageProductContext(;
            fold_strategy = MessagesProductFromLeftToRight(),
            callbacks = handler,
        )

        msg1 = Message(Normal(0, 1), false, false)
        msg2 = Message(Normal(0, 2), false, false)

        result = @inferred(
            compute_product_of_messages(testvar, context, [msg1, msg2])
        )

        @test length(handler.events) == 2

        # Before callback: variable, context, left, right
        before = handler.events[1]
        @test before.event === :before_product_of_two_messages
        @test before.data.variable === testvar
        @test before.data.context === context
        @test getdata(before.data.left) == Normal(0, 1)
        @test getdata(before.data.right) == Normal(0, 2)

        # After callback: variable, context, left, right, result, annotations
        after = handler.events[2]
        @test after.event === :after_product_of_two_messages
        @test after.data.variable === testvar
        @test after.data.context === context
        @test getdata(after.data.left) == Normal(0, 1)
        @test getdata(after.data.right) == Normal(0, 2)
        @test after.data.result == result
        @test after.data.annotations isa AnnotationDict  # AnnotationDict is default
    end
end

@testitem "Form constraint callbacks with FormConstraintCheckEach" setup = [
    MessageProductContextUtils
] begin
    import ReactiveMP:
        MessageProductContext,
        Message,
        FormConstraintCheckEach,
        compute_product_of_messages,
        compute_product_of_two_messages,
        getdata

    messages = [
        Message(Normal(0, 1), false, false)
        Message(Normal(0, 2), false, false)
        Message(Normal(0, 3), false, false)
    ]

    @testset "CheckEach applies form constraint after each pairwise product" begin
        listen_to = (
            :before_form_constraint_applied, :after_form_constraint_applied
        )
        handler = SaveOrderOfComputationCallbacks(listen_to, [])
        context = MessageProductContext(;
            form_constraint = AddOneToMeanConstraint(),
            form_constraint_check_strategy = FormConstraintCheckEach(),
            callbacks = handler,
        )

        result = compute_product_of_messages(testvar, context, messages)

        # Hand-computed expected values (left-to-right fold with +1 to mean after each product):
        # Step 1: prod(Normal(0,1), Normal(0,2))
        #   var = 1/(1 + 1/2) = 2/3, mean = (2/3)*(0 + 0) = 0 => Normal(0, 2/3)
        #   constraint: Normal(0 + 1, 2/3) = Normal(1, 2/3)
        # Step 2: prod(Normal(1, 2/3), Normal(0,3))
        #   var = 1/(3/2 + 1/3) = 6/11, mean = (6/11)*(1*3/2 + 0*1/3) = 9/11 => Normal(9/11, 6/11)
        #   constraint: Normal(9/11 + 1, 6/11) = Normal(20/11, 6/11)
        @test getdata(result) ≈ Normal(20 / 11, 6 / 11)

        # With CheckEach and 3 messages: 2 pairwise products => 2 before + 2 after = 4 events
        @test length(handler.events) == 4
        @test handler.events[1].event === :before_form_constraint_applied
        @test handler.events[2].event === :after_form_constraint_applied
        @test handler.events[3].event === :before_form_constraint_applied
        @test handler.events[4].event === :after_form_constraint_applied

        # All form constraint events should carry the CheckEach strategy
        for e in handler.events
            @test e.data.strategy === FormConstraintCheckEach()
        end

        # First constraint: before gets Normal(0, 2/3), after gets Normal(1, 2/3)
        @test handler.events[1].data.distribution ≈ Normal(0, 2 / 3)
        @test handler.events[2].data.result ≈ Normal(1, 2 / 3)

        # Second constraint: before gets Normal(9/11, 6/11), after gets Normal(20/11, 6/11)
        @test handler.events[3].data.distribution ≈ Normal(9 / 11, 6 / 11)
        @test handler.events[4].data.result ≈ Normal(20 / 11, 6 / 11)
    end
end

@testitem "Form constraint callbacks with FormConstraintCheckLast" setup = [
    MessageProductContextUtils
] begin
    import ReactiveMP:
        MessageProductContext,
        Message,
        FormConstraintCheckLast,
        compute_product_of_messages,
        getdata

    messages = [
        Message(Normal(0, 1), false, false)
        Message(Normal(0, 2), false, false)
        Message(Normal(0, 3), false, false)
    ]

    @testset "CheckLast applies form constraint once at the end" begin
        listen_to = (
            :before_form_constraint_applied, :after_form_constraint_applied
        )
        handler = SaveOrderOfComputationCallbacks(listen_to, [])
        context = MessageProductContext(;
            form_constraint = AddOneToMeanConstraint(),
            form_constraint_check_strategy = FormConstraintCheckLast(),
            callbacks = handler,
        )

        result = compute_product_of_messages(testvar, context, messages)

        # Hand-computed expected values (left-to-right fold, constraint only at the end):
        # Step 1: prod(Normal(0,1), Normal(0,2))
        #   var = 2/3, mean = 0 => Normal(0, 2/3) — no constraint
        # Step 2: prod(Normal(0, 2/3), Normal(0,3))
        #   var = 1/(3/2 + 1/3) = 6/11, mean = 0 => Normal(0, 6/11) — no constraint
        # Final constraint: Normal(0 + 1, 6/11) = Normal(1, 6/11)
        @test getdata(result) ≈ Normal(1, 6 / 11)

        # With CheckLast, form constraint fires only once (1 before + 1 after)
        @test length(handler.events) == 2
        @test handler.events[1].event === :before_form_constraint_applied
        @test handler.events[2].event === :after_form_constraint_applied

        # The strategy should be CheckLast
        @test handler.events[1].data.strategy === FormConstraintCheckLast()
        @test handler.events[2].data.strategy === FormConstraintCheckLast()

        # Before constraint: Normal(0, 6/11), after constraint: Normal(1, 6/11)
        @test handler.events[1].data.distribution ≈ Normal(0, 6 / 11)
        @test handler.events[2].data.result ≈ Normal(1, 6 / 11)

        # The final result matches the after-constraint distribution
        @test getdata(result) ≈ handler.events[2].data.result
    end

    @testset "Before/after product of messages callbacks fire around the whole computation" begin
        listen_to = (:before_product_of_messages, :after_product_of_messages)
        handler = SaveOrderOfComputationCallbacks(listen_to, [])
        context = MessageProductContext(;
            form_constraint = AddOneToMeanConstraint(),
            form_constraint_check_strategy = FormConstraintCheckLast(),
            callbacks = handler,
        )

        result = compute_product_of_messages(testvar, context, messages)

        # BeforeProductOfMessages should be the first event
        @test length(handler.events) == 2
        @test handler.events[1].event === :before_product_of_messages
        @test handler.events[1].data.variable === testvar
        @test handler.events[1].data.context === context
        @test handler.events[1].data.messages === messages

        # AfterProductOfMessages should be the last event
        @test handler.events[2].event === :after_product_of_messages
        @test handler.events[2].data.variable === testvar
        @test handler.events[2].data.context === context
        @test handler.events[2].data.messages === messages
        @test handler.events[2].data.result == result
    end
end

@testitem "Before/after product of messages callbacks" setup = [
    MessageProductContextUtils
] begin
    import ReactiveMP:
        MessageProductContext, Message, compute_product_of_messages, getdata

    @testset "Fires with default context (no form constraint)" begin
        listen_to = (:before_product_of_messages, :after_product_of_messages)
        handler = SaveOrderOfComputationCallbacks(listen_to, [])
        context = MessageProductContext(; callbacks = handler)

        messages = [
            Message(Normal(0, 1), false, false)
            Message(Normal(0, 2), false, false)
            Message(Normal(0, 3), false, false)
        ]

        result = compute_product_of_messages(testvar, context, messages)

        # prod(Normal(0,1), Normal(0,2)) = Normal(0, 2/3)
        # prod(Normal(0, 2/3), Normal(0,3)) = Normal(0, 6/11)
        @test getdata(result) ≈ Normal(0, 6 / 11)

        @test length(handler.events) == 2

        # Before: receives variable, context, and the original messages
        @test handler.events[1].event === :before_product_of_messages
        @test handler.events[1].data.variable === testvar
        @test handler.events[1].data.context === context
        @test handler.events[1].data.messages === messages

        # After: receives variable, context, original messages, and the final result
        @test handler.events[2].event === :after_product_of_messages
        @test handler.events[2].data.variable === testvar
        @test handler.events[2].data.context === context
        @test handler.events[2].data.messages === messages
        @test getdata(handler.events[2].data.result) ≈ getdata(result)
    end

    @testset "Fires with two messages" begin
        listen_to = (:before_product_of_messages, :after_product_of_messages)
        handler = SaveOrderOfComputationCallbacks(listen_to, [])
        context = MessageProductContext(; callbacks = handler)

        messages = [
            Message(Normal(1, 4), false, false)
            Message(Normal(3, 4), false, false)
        ]

        result = compute_product_of_messages(testvar, context, messages)

        # prod(Normal(1,4), Normal(3,4)):
        #   var = 1/(1/4 + 1/4) = 2, mean = 2*(1/4 + 3/4) = 2
        @test getdata(result) ≈ Normal(2, 2)

        @test length(handler.events) == 2
        @test handler.events[1].event === :before_product_of_messages
        @test handler.events[2].event === :after_product_of_messages
        @test getdata(handler.events[2].data.result) ≈ getdata(result)
    end

    @testset "Before fires before any pairwise product, after fires after all" begin
        # Listen to all product-related events to verify ordering
        listen_to = (
            :before_product_of_messages,
            :before_product_of_two_messages,
            :after_product_of_two_messages,
            :after_product_of_messages,
        )
        handler = SaveOrderOfComputationCallbacks(listen_to, [])
        context = MessageProductContext(; callbacks = handler)

        messages = [
            Message(Normal(0, 1), false, false)
            Message(Normal(0, 2), false, false)
            Message(Normal(0, 3), false, false)
        ]

        result = compute_product_of_messages(testvar, context, messages)

        # 3 messages => 1 before_all + 2*(before_two + after_two) + 1 after_all = 6 events
        @test length(handler.events) == 6
        @test handler.events[1].event === :before_product_of_messages
        @test handler.events[2].event === :before_product_of_two_messages
        @test handler.events[3].event === :after_product_of_two_messages
        @test handler.events[4].event === :before_product_of_two_messages
        @test handler.events[5].event === :after_product_of_two_messages
        @test handler.events[6].event === :after_product_of_messages

        # The after_product_of_messages result should match the final result
        @test getdata(handler.events[6].data.result) ≈ getdata(result)
    end
end
