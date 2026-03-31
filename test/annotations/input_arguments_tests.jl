@testmodule RuleInputArgumentsTestUtils begin
    import ReactiveMP:
        AnnotationDict,
        InputArgumentsAnnotations,
        RuleInputArgumentsRecord,
        ProductInputArgumentsRecord

    struct MockMapping
        name::Symbol
    end
end

@testitem "post_rule_annotations! stores a RuleInputArgumentsRecord" setup=[
    RuleInputArgumentsTestUtils
] begin
    import ReactiveMP:
        AnnotationDict,
        post_rule_annotations!,
        InputArgumentsAnnotations,
        RuleInputArgumentsRecord,
        get_rule_input_arguments

    ann       = AnnotationDict()
    mapping   = RuleInputArgumentsTestUtils.MockMapping(:out)
    messages  = (:msg1, :msg2)
    marginals = (:mar1,)
    result    = :the_result

    post_rule_annotations!(
        InputArgumentsAnnotations(), ann, mapping, messages, marginals, result
    )

    record = get_rule_input_arguments(ann)
    @test record isa RuleInputArgumentsRecord
    @test record.mapping === mapping
    @test record.messages === messages
    @test record.marginals === marginals
    @test record.result === result
end

@testitem "post_product_annotations! merges two RuleInputArgumentsRecord into ProductInputArgumentsRecord" setup=[
    RuleInputArgumentsTestUtils
] begin
    import ReactiveMP:
        AnnotationDict,
        annotate!,
        post_product_annotations!,
        InputArgumentsAnnotations,
        RuleInputArgumentsRecord,
        ProductInputArgumentsRecord,
        get_rule_input_arguments

    left_record  = RuleInputArgumentsRecord(RuleInputArgumentsTestUtils.MockMapping(:left), nothing, nothing, :left_result)
    right_record = RuleInputArgumentsRecord(RuleInputArgumentsTestUtils.MockMapping(:right), nothing, nothing, :right_result)

    left_ann  = AnnotationDict()
    right_ann = AnnotationDict()
    annotate!(left_ann, :rule_input_arguments, left_record)
    annotate!(right_ann, :rule_input_arguments, right_record)

    merged = post_product_annotations!(
        (InputArgumentsAnnotations(),),
        left_ann,
        right_ann,
        nothing,
        nothing,
        nothing,
    )

    prod = get_rule_input_arguments(merged)
    @test prod isa ProductInputArgumentsRecord
    @test length(prod.mappings) == 2
    @test prod.mappings[1] === left_record
    @test prod.mappings[2] === right_record
end

@testitem "post_product_annotations! merges record (left) and prod (right)" setup=[
    RuleInputArgumentsTestUtils
] begin
    import ReactiveMP:
        AnnotationDict,
        annotate!,
        post_product_annotations!,
        InputArgumentsAnnotations,
        RuleInputArgumentsRecord,
        ProductInputArgumentsRecord,
        get_rule_input_arguments

    r1 = RuleInputArgumentsRecord(
        RuleInputArgumentsTestUtils.MockMapping(:r1), nothing, nothing, :res1
    )
    r2 = RuleInputArgumentsRecord(
        RuleInputArgumentsTestUtils.MockMapping(:r2), nothing, nothing, :res2
    )
    r3 = RuleInputArgumentsRecord(
        RuleInputArgumentsTestUtils.MockMapping(:r3), nothing, nothing, :res3
    )

    left_ann  = AnnotationDict()
    right_ann = AnnotationDict()
    annotate!(left_ann, :rule_input_arguments, r1)
    annotate!(
        right_ann, :rule_input_arguments, ProductInputArgumentsRecord([r2, r3])
    )

    merged = post_product_annotations!(
        (InputArgumentsAnnotations(),),
        left_ann,
        right_ann,
        nothing,
        nothing,
        nothing,
    )

    prod = get_rule_input_arguments(merged)
    @test prod isa ProductInputArgumentsRecord
    @test length(prod.mappings) == 3
    @test prod.mappings[1] === r1
    @test prod.mappings[2] === r2
    @test prod.mappings[3] === r3
end

@testitem "post_product_annotations! merges prod (left) and record (right)" setup=[
    RuleInputArgumentsTestUtils
] begin
    import ReactiveMP:
        AnnotationDict,
        annotate!,
        post_product_annotations!,
        InputArgumentsAnnotations,
        RuleInputArgumentsRecord,
        ProductInputArgumentsRecord,
        get_rule_input_arguments

    r1 = RuleInputArgumentsRecord(
        RuleInputArgumentsTestUtils.MockMapping(:r1), nothing, nothing, :res1
    )
    r2 = RuleInputArgumentsRecord(
        RuleInputArgumentsTestUtils.MockMapping(:r2), nothing, nothing, :res2
    )
    r3 = RuleInputArgumentsRecord(
        RuleInputArgumentsTestUtils.MockMapping(:r3), nothing, nothing, :res3
    )

    left_ann  = AnnotationDict()
    right_ann = AnnotationDict()
    annotate!(
        left_ann, :rule_input_arguments, ProductInputArgumentsRecord([r1, r2])
    )
    annotate!(right_ann, :rule_input_arguments, r3)

    merged = post_product_annotations!(
        (InputArgumentsAnnotations(),),
        left_ann,
        right_ann,
        nothing,
        nothing,
        nothing,
    )

    prod = get_rule_input_arguments(merged)
    @test prod isa ProductInputArgumentsRecord
    @test length(prod.mappings) == 3
    @test prod.mappings[1] === r1
    @test prod.mappings[2] === r2
    @test prod.mappings[3] === r3
end

@testitem "post_product_annotations! merges two ProductInputArgumentsRecord" setup=[
    RuleInputArgumentsTestUtils
] begin
    import ReactiveMP:
        AnnotationDict,
        annotate!,
        post_product_annotations!,
        InputArgumentsAnnotations,
        RuleInputArgumentsRecord,
        ProductInputArgumentsRecord,
        get_rule_input_arguments

    r1 = RuleInputArgumentsRecord(
        RuleInputArgumentsTestUtils.MockMapping(:r1), nothing, nothing, :res1
    )
    r2 = RuleInputArgumentsRecord(
        RuleInputArgumentsTestUtils.MockMapping(:r2), nothing, nothing, :res2
    )
    r3 = RuleInputArgumentsRecord(
        RuleInputArgumentsTestUtils.MockMapping(:r3), nothing, nothing, :res3
    )
    r4 = RuleInputArgumentsRecord(
        RuleInputArgumentsTestUtils.MockMapping(:r4), nothing, nothing, :res4
    )

    left_ann  = AnnotationDict()
    right_ann = AnnotationDict()
    annotate!(
        left_ann, :rule_input_arguments, ProductInputArgumentsRecord([r1, r2])
    )
    annotate!(
        right_ann, :rule_input_arguments, ProductInputArgumentsRecord([r3, r4])
    )

    merged = post_product_annotations!(
        (InputArgumentsAnnotations(),),
        left_ann,
        right_ann,
        nothing,
        nothing,
        nothing,
    )

    prod = get_rule_input_arguments(merged)
    @test prod isa ProductInputArgumentsRecord
    @test length(prod.mappings) == 4
    @test prod.mappings[1] === r1
    @test prod.mappings[2] === r2
    @test prod.mappings[3] === r3
    @test prod.mappings[4] === r4
end
