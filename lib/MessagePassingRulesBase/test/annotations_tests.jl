@testitem "annotations:sink" tags = [:base] begin
    using MessagePassingRulesBase: AnnotationStore, NoAnnotations, annotate!, getannotation, hasannotation

    s = AnnotationStore()
    @test !hasannotation(s, :logscale)
    @test getannotation(s, :logscale, nothing) === nothing
    annotate!(s, :logscale, -1.5)
    @test hasannotation(s, :logscale)
    @test getannotation(s, :logscale) == -1.5
    @test_throws KeyError getannotation(AnnotationStore(), :logscale)

    @test sizeof(NoAnnotations()) == 0
    @test annotate!(NoAnnotations(), :logscale, 1.0) === nothing
    @test !hasannotation(NoAnnotations(), :logscale)
end

@testitem "annotations:two-way" tags = [:base] begin
    using MessagePassingRulesBase: RuleAnnotations, AnnotationStore, NoAnnotations, annotate!, getannotation

    incoming_out = AnnotationStore()
    annotate!(incoming_out, :logscale, -0.25)

    ann = RuleAnnotations(m = (out = incoming_out,), out = AnnotationStore())
    @test getannotation(ann.m[:out], :logscale) == -0.25

    annotate!(ann, :logscale, 3.0)
    @test getannotation(ann, :logscale) == 3.0
    # Writing the outgoing sink never touches what arrived.
    @test getannotation(ann.m[:out], :logscale) == -0.25

    # No annotations at all costs nothing.
    none = RuleAnnotations()
    @test annotate!(none, :logscale, 1.0) === nothing
    @test sizeof(none) == 0
end
