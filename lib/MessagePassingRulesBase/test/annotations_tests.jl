@testitem "annotations:sink" tags = [:base] begin
    using MessagePassingRulesBase: AnnotationStore, NoAnnotations, annotate!, getannotation, hasannotation

    s = AnnotationStore()
    @test !hasannotation(s, :note)
    @test getannotation(s, :note, nothing) === nothing
    annotate!(s, :note, -1.5)
    @test hasannotation(s, :note)
    @test getannotation(s, :note) == -1.5
    @test_throws KeyError getannotation(AnnotationStore(), :note)

    @test sizeof(NoAnnotations()) == 0
    @test annotate!(NoAnnotations(), :note, 1.0) === nothing
    @test !hasannotation(NoAnnotations(), :note)
end

@testitem "annotations:two-way" tags = [:base] begin
    using MessagePassingRulesBase: RuleAnnotations, AnnotationStore, NoAnnotations, annotate!, getannotation

    incoming_out = AnnotationStore()
    annotate!(incoming_out, :note, -0.25)

    ann = RuleAnnotations(m = (out = incoming_out,), out = AnnotationStore())
    @test getannotation(ann.m[:out], :note) == -0.25

    annotate!(ann, :note, 3.0)
    @test getannotation(ann, :note) == 3.0
    # Writing the outgoing sink never touches what arrived.
    @test getannotation(ann.m[:out], :note) == -0.25

    # No annotations at all costs nothing.
    none = RuleAnnotations()
    @test annotate!(none, :note, 1.0) === nothing
    @test sizeof(none) == 0
end
