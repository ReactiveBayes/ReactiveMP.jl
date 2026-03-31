@testitem "AnnotationDict can be created" begin
    import ReactiveMP: AnnotationDict, annotate!, get_annotation, has_annotation

    ann = AnnotationDict()

    @test !has_annotation(ann, :logscale)

    annotate!(ann, :logscale, 1.0)

    @test has_annotation(ann, :logscale)
    @test get_annotation(ann, :logscale) == 1.0
    @test @inferred(get_annotation(ann, Float64, :logscale)) == 1.0
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
