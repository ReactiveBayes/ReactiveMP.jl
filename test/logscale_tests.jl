@testmodule EngineLogScaleUtils begin
    using ReactiveMP, BayesBase, ExponentialFamily, MessagePassingRulesBase
    import MessagePassingRulesBase: Target, DefaultAlgorithm

    # `compute_logscale` of a product of two of these is 10.
    struct Custom end
    BayesBase.prod(::GenericProd, ::Custom, ::Custom) = Custom()
    BayesBase.compute_logscale(::Custom, ::Custom, ::Custom) = 10.0
    # A pair with no `compute_logscale`.
    struct Opaque end
    BayesBase.prod(::GenericProd, ::Opaque, ::Opaque) = Opaque()

    struct Scaling end
    @define_factor_node(node = Scaling, type = Stochastic, interfaces = [:out, :in, :w])
    @define_message_update_rule(node = Scaling, target = :out, args = (m[:in]::NormalMeanVariance,), logscale = -0.5, body = (args) -> args.m[:in])
    # Declares none.
    @define_message_update_rule(node = Scaling, target = :w, args = (m[:in]::NormalMeanVariance,), body = (args) -> args.m[:in])
    # Reads the incoming log scale.
    @define_message_update_rule(
        node = Scaling, target = :in, args = (m[:out]::NormalMeanVariance,), logscale = (args) -> args.logscale.m[:out] + 1, reads_logscale = true,
        body = (args) -> args.m[:out],
    )

    mapping(target, names; logscales = true, rulefallback = nothing) = ReactiveMP.MessageMapping(
        Scaling, Target{target}(), Val(names), nothing, DefaultAlgorithm(), nothing, Scaling(), nothing,
        ReactiveMP.EngineDiagnostics(), nothing, rulefallback, logscales,
    )

    message(data, logscale = nothing) = Message(data, false, false, ReactiveMP.AnnotationDict(), logscale)
end

@testitem "logscale:messages carry it" tags = [:engine] setup = [EngineLogScaleUtils] begin
    using ReactiveMP, ExponentialFamily, MessagePassingRulesBase
    U = EngineLogScaleUtils

    @test getlogscale(U.message(1.0, -2.0)) === -2.0
    @test getlogscale(as_marginal(U.message(NormalMeanVariance(0.0, 1.0), -2.0))) === -2.0
    @test getlogscale(as_message(as_marginal(U.message(NormalMeanVariance(0.0, 1.0), -2.0)))) === -2.0
    # Not tracked: reading it is an error that says how to track it.
    @test_throws "logscales = true" getlogscale(Message(1.0, false, false))
    @test_throws "logscales = true" getlogscale(Marginal(1.0, false, false))
    @test U.message(1.0, -2.0) == Message(1.0, false, false)
    @test contains(repr(U.message(1.0, -2.0)), "logscale = -2.0")
end

@testitem "logscale:rules" tags = [:engine] setup = [EngineLogScaleUtils] begin
    using ReactiveMP, ExponentialFamily, MessagePassingRulesBase
    U = EngineLogScaleUtils
    m = NormalMeanVariance(0.0, 1.0)

    # Tracked: the declared log scale, or an undefined one naming the rule.
    @test getlogscale(U.mapping(:out, (:in,))((U.message(m, 3.0),), nothing)) === -0.5
    undeclared = getlogscale(U.mapping(:w, (:in,))((U.message(m, 3.0),), nothing))
    @test undeclared isa UndefinedLogScale && undeclared.cause === :no_declaration
    # A rule that reads the incoming log scale gets it as `args.logscale.m[...]`.
    @test getlogscale(U.mapping(:in, (:out,))((U.message(m, 3.0),), nothing)) === 4.0
    @test getlogscale(U.mapping(:in, (:out,))((U.message(m, UndefinedLogScale(:initial)),), nothing)) isa UndefinedLogScale

    # Not tracked: no log scale, and a rule that reads them is an error naming the rule.
    @test U.mapping(:out, (:in,); logscales = false)((U.message(m),), nothing).logscale === nothing
    @test_throws "reads the log scales of its inbound messages" U.mapping(:in, (:out,); logscales = false)((U.message(m),), nothing)

    # A message a rule fallback computes has an undefined log scale.
    fallback = (fform, target, args) -> 42.0
    fell = U.mapping(:out, (:in,); rulefallback = fallback)((U.message(1.0, 0.0),), nothing)
    @test getdata(fell) == 42.0 && getlogscale(fell).cause === :fallback
end

@testitem "logscale:missing inputs" tags = [:engine] setup = [EngineLogScaleUtils] begin
    using ReactiveMP, ExponentialFamily, MessagePassingRulesBase
    import ReactiveMP: MessageProductContext, compute_product_of_two_messages, randomvar
    U = EngineLogScaleUtils

    # No rule runs on a missing input, so no log scale is invented.
    deferred = U.mapping(:out, (:in,))((U.message(missing),), nothing)
    @test getdata(deferred) === missing && deferred.logscale === nothing
    # A product with it is the other side, with its log scale, on either side.
    concrete = U.message(NormalMeanVariance(1.0, 2.0), 4.0)
    context = MessageProductContext()
    @test getlogscale(compute_product_of_two_messages(randomvar(), context, deferred, concrete)) == 4.0
    @test getlogscale(compute_product_of_two_messages(randomvar(), context, concrete, deferred)) == 4.0
end

@testitem "logscale:products" tags = [:engine] setup = [EngineLogScaleUtils] begin
    using ReactiveMP, BayesBase, ExponentialFamily, MessagePassingRulesBase
    import ReactiveMP: MessageProductContext, compute_product_of_two_messages, compute_product_of_messages, randomvar,
        FormConstraintCheckEach, FormConstraintCheckLast, AbstractFormConstraint
    U = EngineLogScaleUtils
    struct ToMean <: AbstractFormConstraint end
    ReactiveMP.constrain_form(::ToMean, d) = PointMass(mean(d))
    # A check that changes nothing, as RxInfer's check of a supported form.
    struct Check <: AbstractFormConstraint end
    ReactiveMP.constrain_form(::Check, d) = d
    product(l, r; kwargs...) = compute_product_of_two_messages(randomvar(), MessageProductContext(; kwargs...), l, r)

    # Both sides' log scales and the product's own.
    @test getlogscale(product(U.message(U.Custom(), 1.0), U.message(U.Custom(), 2.0))) == 13.0
    a, b = NormalMeanVariance(0.0, 1.0), NormalMeanVariance(1.0, 2.0)
    @test getlogscale(product(U.message(a, 0.0), U.message(b, 0.0))) ≈ logpdf(NormalMeanVariance(0.0, 3.0), 1.0)
    # An undefined side propagates, keeping its reason.
    @test getlogscale(product(U.message(U.Custom(), UndefinedLogScale(:initial)), U.message(U.Custom(), 2.0))).cause === :initial
    @test getlogscale(product(U.message(U.Custom(), 2.0), U.message(U.Custom(), UndefinedLogScale(:fallback)))).cause === :fallback
    # Not tracked on either side: none.
    @test product(U.message(U.Custom()), U.message(U.Custom(), 2.0)).logscale === nothing
    @test product(U.message(U.Custom(), 2.0), U.message(U.Custom())).logscale === nothing
    # A pair with no `compute_logscale`.
    opaque = getlogscale(product(U.message(U.Opaque(), 0.0), U.message(U.Opaque(), 0.0)))
    @test opaque.cause === :no_compute_logscale && opaque.detail == (U.Opaque, U.Opaque)
    # A form constraint changes the product, under either strategy.
    constrained = product(U.message(a, 0.0), U.message(b, 0.0); form_constraint = ToMean(), form_constraint_check_strategy = FormConstraintCheckEach())
    @test getlogscale(constrained).cause === :form_constraint
    last = compute_product_of_messages(
        randomvar(), MessageProductContext(; form_constraint = ToMean(), form_constraint_check_strategy = FormConstraintCheckLast()),
        (U.message(a, 0.0), U.message(b, 0.0)),
    )
    @test getlogscale(last).cause === :form_constraint
    # One that returns the product unchanged keeps its log scale.
    unchanged = product(U.message(a, 0.0), U.message(b, 0.0); form_constraint = Check(), form_constraint_check_strategy = FormConstraintCheckEach())
    @test getlogscale(unchanged) ≈ logpdf(NormalMeanVariance(0.0, 3.0), 1.0)
end

@testitem "logscale:variables" tags = [:engine] begin
    using ReactiveMP, BayesBase, Rocket
    import ReactiveMP: MessageObservable, set_initial_message!

    # An observation and a constant are point masses: zero.
    received = Ref{Any}(nothing)
    subscribe!(ReactiveMP.get_stream_of_marginals(constvar(1.0)), (marginal) -> received[] = marginal)
    @test getlogscale(received[]) === 0
    # An initial message no rule computed: undefined.
    stream = MessageObservable()
    set_initial_message!(stream, 1.0)
    @test getlogscale(Rocket.getrecent(stream)).cause === :initial
end
