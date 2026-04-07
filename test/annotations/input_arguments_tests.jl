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

@testitem "Base.show for RuleInputArgumentsRecord" begin
    import ReactiveMP: RuleInputArgumentsRecord, MessageMapping, Marginalisation
    import BayesBase: PointMass

    struct ShowRecordNode end

    mapping = MessageMapping(
        ShowRecordNode,
        Val(:out),
        Marginalisation(),
        Val((:in1, :in2)),
        Val((:q1,)),
        "some-meta",
        nothing,
        ShowRecordNode(),
        nothing,
        nothing,
    )

    record = RuleInputArgumentsRecord(
        mapping, (PointMass(1.0), 2.0), (10.0,), 42.0
    )

    output = sprint(show, record)

    @test occursin("Rule input arguments:", output)
    @test occursin("node:", output)
    @test occursin("ShowRecordNode", output)
    @test occursin("interface:", output)
    @test occursin(":out", output)
    @test occursin("constraint:", output)
    @test occursin("Marginalisation", output)
    @test occursin("meta:", output)
    @test occursin("some-meta", output)
    @test occursin("msg(in1) = BayesBase.PointMass{Float64}(1.0)", output)
    @test occursin("msg(in2) = 2.0", output)
    @test occursin("q(q1) = 10.0", output)
    @test occursin("result:", output)
    @test occursin("42.0", output)
end

@testitem "Base.show for RuleInputArgumentsRecord skips meta when nothing" begin
    import ReactiveMP: RuleInputArgumentsRecord, MessageMapping, Marginalisation

    struct ShowRecordNoMetaNode end

    mapping = MessageMapping(
        ShowRecordNoMetaNode,
        Val(:out),
        Marginalisation(),
        Val((:in,)),
        nothing,
        nothing,
        nothing,
        ShowRecordNoMetaNode(),
        nothing,
        nothing,
    )

    record = RuleInputArgumentsRecord(mapping, (1.0,), nothing, 2.0)
    output = sprint(show, record)

    @test !occursin("meta:", output)
    @test occursin("msg(in) = 1.0", output)
end

@testitem "Base.show for RuleInputArgumentsRecord skips messages/marginals when nothing" begin
    import ReactiveMP: RuleInputArgumentsRecord, MessageMapping, Marginalisation

    struct ShowRecordEmptyInputsNode end

    mapping = MessageMapping(
        ShowRecordEmptyInputsNode,
        Val(:out),
        Marginalisation(),
        nothing,
        nothing,
        nothing,
        nothing,
        ShowRecordEmptyInputsNode(),
        nothing,
        nothing,
    )

    record = RuleInputArgumentsRecord(mapping, nothing, nothing, :the_result)
    output = sprint(show, record)

    @test !occursin("msg(", output)
    @test !occursin("q(", output)
    @test occursin("result:", output)
    @test occursin("the_result", output)
end

@testitem "Base.show for ProductInputArgumentsRecord" begin
    import ReactiveMP:
        RuleInputArgumentsRecord,
        ProductInputArgumentsRecord,
        MessageMapping,
        Marginalisation

    struct ShowProductNodeA end
    struct ShowProductNodeB end

    mapping_a = MessageMapping(
        ShowProductNodeA,
        Val(:out),
        Marginalisation(),
        Val((:in,)),
        nothing,
        nothing,
        nothing,
        ShowProductNodeA(),
        nothing,
        nothing,
    )

    mapping_b = MessageMapping(
        ShowProductNodeB,
        Val(:mean),
        Marginalisation(),
        Val((:x,)),
        nothing,
        nothing,
        nothing,
        ShowProductNodeB(),
        nothing,
        nothing,
    )

    r1 = RuleInputArgumentsRecord(mapping_a, (1.0,), nothing, :res_a)
    r2 = RuleInputArgumentsRecord(mapping_b, (2.0,), nothing, :res_b)
    prod = ProductInputArgumentsRecord([r1, r2])

    output = sprint(show, prod)

    @test occursin("Product of 2 rule input arguments:", output)
    @test occursin("[1]", output)
    @test occursin("[2]", output)
    @test occursin("ShowProductNodeA", output)
    @test occursin("ShowProductNodeB", output)
    @test occursin("res_a", output)
    @test occursin("res_b", output)
end

@testitem "AddonMemory throws an error" begin
    import ReactiveMP: AddonMemory

    @test_throws "AddonMemory` has been removed" AddonMemory()
    @test_throws "InputArgumentsAnnotations" AddonMemory()
end
