@testmodule LogScaleAnnotationsTestUtils begin
    import ReactiveMP: AnnotationDict, LogScaleAnnotations
    import BayesBase: compute_logscale, PointMass
    import BayesBase

    struct CustomDistributionForLogScaleTesting end

    BayesBase.compute_logscale(
        ::CustomDistributionForLogScaleTesting,
        ::CustomDistributionForLogScaleTesting,
        ::CustomDistributionForLogScaleTesting,
    ) = 10.0
end

@testitem "getlogscale reads from AnnotationDict" begin
    import ReactiveMP: AnnotationDict, annotate!, getlogscale

    ann = AnnotationDict()
    annotate!(ann, :logscale, 3.0)

    @test getlogscale(ann) == 3.0
end

@testitem "getlogscale throws when logscale is not set" begin
    import ReactiveMP: AnnotationDict, getlogscale

    ann = AnnotationDict()

    @test_throws KeyError getlogscale(ann)
end

@testitem "@logscale macro sets logscale annotation via getannotations" begin
    import ReactiveMP: AnnotationDict, getlogscale, @logscale

    _annotations = AnnotationDict()
    getannotations = () -> _annotations
    @logscale 2.5

    @test getlogscale(_annotations) == 2.5
end

@testitem "post_rule_annotations! is no-op when logscale already annotated" setup = [
    LogScaleAnnotationsTestUtils
] begin
    import ReactiveMP:
        AnnotationDict,
        annotate!,
        getlogscale,
        post_rule_annotations!,
        LogScaleAnnotations

    ann = AnnotationDict()
    annotate!(ann, :logscale, 7.0)

    post_rule_annotations!(
        LogScaleAnnotations(), ann, nothing, nothing, nothing, nothing
    )

    @test getlogscale(ann) == 7.0
end

@testitem "post_rule_annotations! sets logscale to 0 when all messages are PointMass" setup = [
    LogScaleAnnotationsTestUtils
] begin
    import ReactiveMP:
        AnnotationDict,
        getlogscale,
        post_rule_annotations!,
        LogScaleAnnotations,
        Message
    import BayesBase: PointMass

    ann      = AnnotationDict()
    messages = (Message(PointMass(1.0), false, false), Message(PointMass(2.0), false, false))

    post_rule_annotations!(
        LogScaleAnnotations(), ann, nothing, messages, nothing, nothing
    )

    @test getlogscale(ann) == 0
end

@testitem "post_rule_annotations! sets logscale to 0 when all marginals are PointMass" setup = [
    LogScaleAnnotationsTestUtils
] begin
    import ReactiveMP:
        AnnotationDict,
        getlogscale,
        post_rule_annotations!,
        LogScaleAnnotations,
        Marginal
    import BayesBase: PointMass

    ann       = AnnotationDict()
    marginals = (Marginal(PointMass(1.0), false, false),)

    post_rule_annotations!(
        LogScaleAnnotations(), ann, nothing, nothing, marginals, nothing
    )

    @test getlogscale(ann) == 0
end

@testitem "post_rule_annotations! errors when logscale not set and inputs are not all PointMass" setup = [
    LogScaleAnnotationsTestUtils
] begin
    import ReactiveMP:
        AnnotationDict, post_rule_annotations!, LogScaleAnnotations

    ann      = AnnotationDict()
    messages = (Message(LogScaleAnnotationsTestUtils.CustomDistributionForLogScaleTesting(), false, false),)

    @test_throws "Log-scale annotation has not been set" post_rule_annotations!(
        LogScaleAnnotations(), ann, nothing, messages, nothing, nothing
    )
end

@testitem "post_product_annotations! with LogScaleAnnotations sums logscales and adds compute_logscale" setup = [
    LogScaleAnnotationsTestUtils
] begin
    import ReactiveMP:
        AnnotationDict,
        annotate!,
        getlogscale,
        post_product_annotations!,
        LogScaleAnnotations

    left_ann  = AnnotationDict()
    right_ann = AnnotationDict()
    annotate!(left_ann, :logscale, 1.0)
    annotate!(right_ann, :logscale, 2.0)

    dist   = LogScaleAnnotationsTestUtils.CustomDistributionForLogScaleTesting()
    merged = post_product_annotations!((LogScaleAnnotations(),), left_ann, right_ann, dist, dist, dist)

    # 1.0 + 2.0 + compute_logscale(...) = 1.0 + 2.0 + 10.0 = 13.0
    @test getlogscale(merged) == 13.0
end

@testitem "A `missing` message stays deferred under LogScaleAnnotations" begin
    using ReactiveMP, BayesBase, Distributions, ExponentialFamily

    import ReactiveMP:
        MessageMapping,
        LogScaleAnnotations,
        Message,
        getdata,
        getannotations,
        has_annotation,
        MessageProductContext,
        compute_product_of_two_messages,
        randomvar,
        activate!,
        RandomVariableActivationOptions

    struct NodeForDeferredLogScaleTest end

    @node NodeForDeferredLogScaleTest Stochastic [out, in]

    # A rule whose input is *not* a `PointMass`, so `LogScaleAnnotations` cannot fall back
    # to `:logscale = 0` and would `error()` if it ran. The rule body itself deliberately
    # never sets `@logscale`.
    @rule NodeForDeferredLogScaleTest(:out, Marginalisation) (
        m_in::NormalMeanVariance,
    ) = NormalMeanVariance(mean(m_in), var(m_in))

    mapping = MessageMapping(
        NodeForDeferredLogScaleTest,
        Val(:out),
        Marginalisation(),
        Val((:in,)),
        nothing,
        nothing,
        (LogScaleAnnotations(),),
        NodeForDeferredLogScaleTest(),
        nothing,
        nothing,
    )

    @testset "a concrete message still goes through the processor" begin
        # Sanity check that the setup is right: with a real (non-PointMass) message and no
        # `@logscale` in the rule body, the processor is reached and does error. This is what
        # makes the `missing` case below meaningful rather than vacuous.
        @test_throws "Log-scale annotation has not been set" mapping(
            (Message(NormalMeanVariance(0.0, 1.0), false, false),), nothing
        )
    end

    @testset "a missing message is returned, not turned into an error" begin
        # `MessageMapping` short-circuits to `missing` without running the rule. Previously
        # the post-rule processors ran anyway, so `LogScaleAnnotations` turned a legitimately
        # deferred message into a hard crash (issue #623).
        result = mapping((Message(missing, false, false),), nothing)

        @test getdata(result) === missing
        # No log-scale was invented for a message that never went through a rule.
        @test !has_annotation(getannotations(result), :logscale)
    end

    @testset "a deferred message survives a subsequent product" begin
        # The deferred message carries no `:logscale`, so the product must not try to read
        # one. `post_product_annotations!`'s `::Missing` dispatches handle this by copying
        # the other side's annotations through unchanged.
        deferred = mapping((Message(missing, false, false),), nothing)

        concrete_ann = ReactiveMP.AnnotationDict()
        ReactiveMP.annotate!(concrete_ann, :logscale, 4.0)
        concrete = Message(
            NormalMeanVariance(1.0, 2.0), false, false, concrete_ann
        )

        variable = randomvar()
        context  = MessageProductContext(annotations = (LogScaleAnnotations(),))

        # Both orderings: the missing side may be on the left or on the right.
        left_product = compute_product_of_two_messages(
            variable, context, deferred, concrete
        )
        right_product = compute_product_of_two_messages(
            variable, context, concrete, deferred
        )

        @test ReactiveMP.getlogscale(getannotations(left_product)) == 4.0
        @test ReactiveMP.getlogscale(getannotations(right_product)) == 4.0
    end
end

@testitem "AddonLogScale throws an error" begin
    import ReactiveMP: AddonLogScale

    @test_throws "AddonLogScale` has been removed" AddonLogScale()
    @test_throws "LogScaleAnnotations" AddonLogScale()
end
