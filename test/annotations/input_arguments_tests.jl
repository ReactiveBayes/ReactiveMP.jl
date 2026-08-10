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

@testitem "post_product_annotations! copies the right record through when the left side never ran a rule (e.g. a clamped constant)" setup=[
    RuleInputArgumentsTestUtils
] begin
    import ReactiveMP:
        AnnotationDict,
        annotate!,
        post_product_annotations!,
        InputArgumentsAnnotations,
        RuleInputArgumentsRecord,
        get_rule_input_arguments,
        has_annotation

    right_record = RuleInputArgumentsRecord(
        RuleInputArgumentsTestUtils.MockMapping(:right),
        nothing,
        nothing,
        :right_result,
    )

    left_ann  = AnnotationDict() # empty: represents a clamped/constant message, which never runs a rule
    right_ann = AnnotationDict()
    annotate!(right_ann, :rule_input_arguments, right_record)

    merged = post_product_annotations!(
        (InputArgumentsAnnotations(),),
        left_ann,
        right_ann,
        nothing,
        nothing,
        nothing,
    )

    @test has_annotation(merged, :rule_input_arguments)
    @test get_rule_input_arguments(merged) === right_record
end

@testitem "post_product_annotations! copies the left record through when the right side never ran a rule (e.g. a clamped constant)" setup=[
    RuleInputArgumentsTestUtils
] begin
    import ReactiveMP:
        AnnotationDict,
        annotate!,
        post_product_annotations!,
        InputArgumentsAnnotations,
        RuleInputArgumentsRecord,
        get_rule_input_arguments,
        has_annotation

    left_record = RuleInputArgumentsRecord(
        RuleInputArgumentsTestUtils.MockMapping(:left),
        nothing,
        nothing,
        :left_result,
    )

    left_ann  = AnnotationDict()
    right_ann = AnnotationDict() # empty: represents a clamped/constant message, which never runs a rule
    annotate!(left_ann, :rule_input_arguments, left_record)

    merged = post_product_annotations!(
        (InputArgumentsAnnotations(),),
        left_ann,
        right_ann,
        nothing,
        nothing,
        nothing,
    )

    @test has_annotation(merged, :rule_input_arguments)
    @test get_rule_input_arguments(merged) === left_record
end

@testitem "post_product_annotations! leaves the merged annotation empty when neither side ran a rule (product of two clamped constants)" setup=[
    RuleInputArgumentsTestUtils
] begin
    import ReactiveMP:
        AnnotationDict,
        post_product_annotations!,
        InputArgumentsAnnotations,
        has_annotation

    left_ann  = AnnotationDict()
    right_ann = AnnotationDict()

    merged = post_product_annotations!(
        (InputArgumentsAnnotations(),),
        left_ann,
        right_ann,
        nothing,
        nothing,
        nothing,
    )

    @test !has_annotation(merged, :rule_input_arguments)
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

@testitem "merging does not mutate the operand records" setup = [
    RuleInputArgumentsTestUtils
] begin
    # `EqualityChain` hands the *same* cached `Message` -- and therefore the same
    # `AnnotationDict` and the same `ProductInputArgumentsRecord` -- to several
    # all-but-one outbound products. When `_merge_input_arguments` merged by
    # `push!`/`append!`/`pushfirst!`-ing into one operand's `mappings` vector and
    # returning that same object, the second product observed a record that had
    # already grown from the first, so the stored trace accumulated rule executions
    # that never contributed to it (issue #622).
    #
    # These tests merge the same operand twice and assert the operand is unchanged
    # and that the two results are independent, which is the property the
    # copy-on-write implementation provides and the in-place one did not.
    import ReactiveMP:
        AnnotationDict,
        annotate!,
        post_product_annotations!,
        InputArgumentsAnnotations,
        RuleInputArgumentsRecord,
        ProductInputArgumentsRecord,
        get_rule_input_arguments,
        _merge_input_arguments

    record(name) = RuleInputArgumentsRecord(
        RuleInputArgumentsTestUtils.MockMapping(name), nothing, nothing, name
    )

    @testset "prod (left) merged twice with different records" begin
        shared = ProductInputArgumentsRecord([record(:a), record(:b)])

        first  = _merge_input_arguments(shared, record(:c))
        second = _merge_input_arguments(shared, record(:d))

        # The shared operand must not have grown.
        @test length(shared.mappings) == 2
        @test map(r -> r.result, shared.mappings) == [:a, :b]

        # Neither result aliases the operand, and neither leaked into the other.
        @test second !== shared
        @test second.mappings !== shared.mappings
        @test map(r -> r.result, first.mappings) == [:a, :b, :c]
        @test map(r -> r.result, second.mappings) == [:a, :b, :d]
    end

    @testset "prod (right) merged twice with different records" begin
        shared = ProductInputArgumentsRecord([record(:a), record(:b)])

        first  = _merge_input_arguments(record(:c), shared)
        second = _merge_input_arguments(record(:d), shared)

        @test length(shared.mappings) == 2
        @test map(r -> r.result, shared.mappings) == [:a, :b]
        @test map(r -> r.result, first.mappings) == [:c, :a, :b]
        @test map(r -> r.result, second.mappings) == [:d, :a, :b]
    end

    @testset "prod merged twice with another prod" begin
        shared = ProductInputArgumentsRecord([record(:a), record(:b)])
        other  = ProductInputArgumentsRecord([record(:c)])

        first  = _merge_input_arguments(shared, other)
        second = _merge_input_arguments(shared, other)

        @test length(shared.mappings) == 2
        @test length(other.mappings) == 1
        @test map(r -> r.result, first.mappings) == [:a, :b, :c]
        @test map(r -> r.result, second.mappings) == [:a, :b, :c]
        @test first.mappings !== second.mappings
    end

    @testset "reusing one side's annotations across two products" begin
        # The same scenario one layer up, through the public processor entry point:
        # a single `left_ann` (as a cached message would supply) feeding two products.
        shared_record = ProductInputArgumentsRecord([record(:a), record(:b)])
        left_ann      = AnnotationDict()
        annotate!(left_ann, :rule_input_arguments, shared_record)

        function merge_with(name)
            right_ann = AnnotationDict()
            annotate!(right_ann, :rule_input_arguments, record(name))
            return get_rule_input_arguments(
                post_product_annotations!(
                    (InputArgumentsAnnotations(),),
                    left_ann,
                    right_ann,
                    nothing,
                    nothing,
                    nothing,
                ),
            )
        end

        first  = merge_with(:c)
        second = merge_with(:d)

        # Under the in-place implementation this was [:a, :b, :c, :d] -- the second
        # product's trace contained `:c`, which never contributed to it.
        @test map(r -> r.result, first.mappings) == [:a, :b, :c]
        @test map(r -> r.result, second.mappings) == [:a, :b, :d]
        @test map(r -> r.result, shared_record.mappings) == [:a, :b]
    end
end

@testitem "AddonMemory throws an error" begin
    import ReactiveMP: AddonMemory

    @test_throws "AddonMemory` has been removed" AddonMemory()
    @test_throws "InputArgumentsAnnotations" AddonMemory()
end
