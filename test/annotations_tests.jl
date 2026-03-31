@testmodule AnnotationsTestUtils begin
    import ReactiveMP: AbstractAnnotations, AnnotationDict, annotate!, get_annotation, post_product_annotations!
    import ReactiveMP

    struct Normal
        mean::Float64
        std::Float64
    end

    struct SumAnnotations <: AbstractAnnotations end

    function ReactiveMP.post_product_annotations!(::SumAnnotations, merged, left_ann, right_ann, new_dist, left_dist, right_dist)
        annotate!(merged, :sum, get_annotation(left_ann, :val) + get_annotation(right_ann, :val))
    end
end

@testitem "AnnotationDict can be created" begin
    import ReactiveMP: AnnotationDict, annotate!, get_annotation, has_annotation

    ann = AnnotationDict()

    @test !has_annotation(ann, :logscale)

    annotate!(ann, :logscale, 1.0)

    @test has_annotation(ann, :logscale)
    @test get_annotation(ann, :logscale) == 1.0
    @test @inferred(get_annotation(ann, Float64, :logscale)) == 1.0
end

@testitem "AnnotationDict can be copied with copy constructor" begin
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

@testitem "AnnotationDict does not allocate on simple creation" begin
    import ReactiveMP: AnnotationDict, has_annotation

    function foo()
        ann = AnnotationDict()
        return has_annotation(ann, :logscale)
    end

    foo()

    @test @allocated(foo()) === 0
end

@testitem "post_product_annotations! with no processors returns empty AnnotationDict" setup=[AnnotationsTestUtils] begin
    import ReactiveMP: AnnotationDict, annotate!, has_annotation, post_product_annotations!

    left_ann  = AnnotationDict()
    right_ann = AnnotationDict()
    annotate!(left_ann,  :foo, 1)
    annotate!(right_ann, :foo, 2)

    dist = AnnotationsTestUtils.Normal(0.0, 1.0)

    for processors in (nothing, ())
        result = post_product_annotations!(processors, left_ann, right_ann, dist, dist, dist)
        @test result isa AnnotationDict
        @test !has_annotation(result, :foo)
    end
end

@testitem "post_product_annotations! calls per-processor post_product_annotations! for each processor" setup=[AnnotationsTestUtils] begin
    import ReactiveMP: AnnotationDict, annotate!, get_annotation, has_annotation, post_product_annotations!

    left_ann  = AnnotationDict()
    right_ann = AnnotationDict()
    annotate!(left_ann,  :val, 3)
    annotate!(right_ann, :val, 7)

    dist = AnnotationsTestUtils.Normal(0.0, 1.0)

    result = post_product_annotations!((AnnotationsTestUtils.SumAnnotations(),), left_ann, right_ann, dist, dist, dist)
    @test has_annotation(result, :sum)
    @test get_annotation(result, :sum) == 10
end

@testitem "post_product_annotations! with missing left_dist copies right_ann" setup=[AnnotationsTestUtils] begin
    import ReactiveMP: AnnotationDict, annotate!, get_annotation, has_annotation, post_product_annotations!

    left_ann  = AnnotationDict()
    right_ann = AnnotationDict()
    annotate!(right_ann, :logscale, 5.0)

    dist = AnnotationsTestUtils.Normal(0.0, 1.0)

    result = post_product_annotations!(nothing, left_ann, right_ann, dist, missing, dist)
    @test has_annotation(result, :logscale)
    @test get_annotation(result, :logscale) == 5.0
end

@testitem "post_product_annotations! with missing right_dist copies left_ann" setup=[AnnotationsTestUtils] begin
    import ReactiveMP: AnnotationDict, annotate!, get_annotation, has_annotation, post_product_annotations!

    left_ann  = AnnotationDict()
    right_ann = AnnotationDict()
    annotate!(left_ann, :logscale, 3.0)

    dist = AnnotationsTestUtils.Normal(0.0, 1.0)

    result = post_product_annotations!(nothing, left_ann, right_ann, dist, dist, missing)
    @test has_annotation(result, :logscale)
    @test get_annotation(result, :logscale) == 3.0
end

@testitem "post_product_annotations! with both dists missing returns empty AnnotationDict" setup=[AnnotationsTestUtils] begin
    import ReactiveMP: AnnotationDict, annotate!, has_annotation, post_product_annotations!

    left_ann  = AnnotationDict()
    right_ann = AnnotationDict()
    annotate!(left_ann,  :logscale, 1.0)
    annotate!(right_ann, :logscale, 2.0)

    result = post_product_annotations!(nothing, left_ann, right_ann, missing, missing, missing)
    @test result isa AnnotationDict
    @test !has_annotation(result, :logscale)
end
