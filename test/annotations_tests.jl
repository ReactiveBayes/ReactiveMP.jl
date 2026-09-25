@testmodule AnnotationsTestUtils begin
    import ReactiveMP:
        AbstractAnnotations,
        AnnotationDict,
        annotate!,
        get_annotation,
        post_product_annotations!
    import ReactiveMP

    struct Normal
        mean::Float64
        std::Float64
    end

    struct SumAnnotations <: AbstractAnnotations end

    function ReactiveMP.post_product_annotations!(
            ::SumAnnotations,
            merged,
            left_ann,
            right_ann,
            new_dist,
            left_dist,
            right_dist,
        )
        annotate!(
            merged,
            :sum,
            get_annotation(left_ann, :val) + get_annotation(right_ann, :val),
        )
    end
end

@testitem "AnnotationDict can be created" tags = [:engine] begin
    import ReactiveMP: AnnotationDict, annotate!, get_annotation, has_annotation

    ann = AnnotationDict()

    @test !has_annotation(ann, :note)

    annotate!(ann, :note, 1.0)

    @test has_annotation(ann, :note)
    @test get_annotation(ann, :note) == 1.0
    @test @inferred(get_annotation(ann, Float64, :note)) == 1.0
end

@testitem "AnnotationDict can be copied with copy constructor" tags = [:engine] begin
    import ReactiveMP: AnnotationDict, annotate!, get_annotation, has_annotation

    original = AnnotationDict()
    annotate!(original, :foo, 1)
    annotate!(original, :bar, 2)

    copied = AnnotationDict(original)

    @test has_annotation(copied, :foo)
    @test has_annotation(copied, :bar)
    @test get_annotation(copied, :foo) == 1
    @test get_annotation(copied, :bar) == 2

    # mutating the copy does not affect the original
    annotate!(copied, :foo, 99)
    @test get_annotation(original, :foo) == 1
end

@testitem "AnnotationDict isempty" tags = [:engine] begin
    import ReactiveMP: AnnotationDict, annotate!

    ann = AnnotationDict()
    @test isempty(ann)

    annotate!(ann, :foo, 1)
    @test !isempty(ann)
end

@testitem "AnnotationDict show" tags = [:engine] begin
    import ReactiveMP: AnnotationDict, annotate!

    ann = AnnotationDict()
    @test repr(ann) == "AnnotationDict()"
    @test sprint(show, ann; context = :compact => true) == "AnnotationDict()"

    annotate!(ann, :note, 1.0)
    # Default form (`:compact => false`) keeps the full key/value listing —
    # this is what an interactive user sees in REPL/Pluto.
    long = repr(ann)
    @test occursin("note", long)
    @test occursin("1.0", long)
    # Compact form (used by trace loggers) collapses to the entry count.
    @test sprint(show, ann; context = :compact => true) == "AnnotationDict(n=1)"

    annotate!(ann, :upper_bound, 2.5)
    @test sprint(show, ann; context = :compact => true) == "AnnotationDict(n=2)"
end

@testitem "AnnotationDict does not allocate on simple creation" tags = [
    :engine, :alloc,
] begin
    import ReactiveMP: AnnotationDict, has_annotation

    function foo()
        ann = AnnotationDict()
        return has_annotation(ann, :note)
    end

    foo()

    @test @allocated(foo()) === 0
end

@testitem "post_product_annotations! with no processors returns empty AnnotationDict" tags = [
    :engine,
] setup = [AnnotationsTestUtils] begin
    import ReactiveMP:
        AnnotationDict, annotate!, has_annotation, post_product_annotations!

    left_ann = AnnotationDict()
    right_ann = AnnotationDict()
    annotate!(left_ann, :foo, 1)
    annotate!(right_ann, :foo, 2)

    dist = AnnotationsTestUtils.Normal(0.0, 1.0)

    for processors in (nothing, ())
        result = post_product_annotations!(
            processors, left_ann, right_ann, dist, dist, dist
        )
        @test result isa AnnotationDict
        @test !has_annotation(result, :foo)
    end
end

@testitem "post_product_annotations! calls per-processor post_product_annotations! for each processor" tags = [
    :engine,
] setup = [AnnotationsTestUtils] begin
    import ReactiveMP:
        AnnotationDict,
        annotate!,
        get_annotation,
        has_annotation,
        post_product_annotations!

    left_ann = AnnotationDict()
    right_ann = AnnotationDict()
    annotate!(left_ann, :val, 3)
    annotate!(right_ann, :val, 7)

    dist = AnnotationsTestUtils.Normal(0.0, 1.0)

    result = post_product_annotations!(
        (AnnotationsTestUtils.SumAnnotations(),),
        left_ann,
        right_ann,
        dist,
        dist,
        dist,
    )
    @test has_annotation(result, :sum)
    @test get_annotation(result, :sum) == 10
end

@testitem "post_product_annotations! with missing left_dist copies right_ann" tags = [
    :engine,
] setup = [AnnotationsTestUtils] begin
    import ReactiveMP:
        AnnotationDict,
        annotate!,
        get_annotation,
        has_annotation,
        post_product_annotations!

    left_ann = AnnotationDict()
    right_ann = AnnotationDict()
    annotate!(right_ann, :note, 5.0)

    dist = AnnotationsTestUtils.Normal(0.0, 1.0)

    result = post_product_annotations!(
        nothing, left_ann, right_ann, dist, missing, dist
    )
    @test has_annotation(result, :note)
    @test get_annotation(result, :note) == 5.0
end

@testitem "post_product_annotations! with missing right_dist copies left_ann" tags = [
    :engine,
] setup = [AnnotationsTestUtils] begin
    import ReactiveMP:
        AnnotationDict,
        annotate!,
        get_annotation,
        has_annotation,
        post_product_annotations!

    left_ann = AnnotationDict()
    right_ann = AnnotationDict()
    annotate!(left_ann, :note, 3.0)

    dist = AnnotationsTestUtils.Normal(0.0, 1.0)

    result = post_product_annotations!(
        nothing, left_ann, right_ann, dist, dist, missing
    )
    @test has_annotation(result, :note)
    @test get_annotation(result, :note) == 3.0
end

@testitem "post_product_annotations! with both dists missing returns empty AnnotationDict" tags = [
    :engine,
] setup = [AnnotationsTestUtils] begin
    import ReactiveMP:
        AnnotationDict, annotate!, has_annotation, post_product_annotations!

    left_ann = AnnotationDict()
    right_ann = AnnotationDict()
    annotate!(left_ann, :note, 1.0)
    annotate!(right_ann, :note, 2.0)

    result = post_product_annotations!(
        nothing, left_ann, right_ann, missing, missing, missing
    )
    @test result isa AnnotationDict
    @test !has_annotation(result, :note)
end
