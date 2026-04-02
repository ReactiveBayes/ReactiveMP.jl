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

@testitem "AddonLogScale throws an error" begin
    import ReactiveMP: AddonLogScale

    @test_throws "AddonLogScale` has been removed" AddonLogScale()
    @test_throws "LogScaleAnnotations" AddonLogScale()
end
