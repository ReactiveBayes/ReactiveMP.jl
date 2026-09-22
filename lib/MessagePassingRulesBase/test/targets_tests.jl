@testitem "targets" tags = [:base] begin
    using MessagePassingRulesBase: Target, IndexedTarget, target_edge, target_index

    @test Target(:out) === Target{:out}()
    @test target_edge(Target(:out)) === :out

    t = IndexedTarget(:m, 3)
    @test t isa IndexedTarget{:m}
    @test target_edge(t) === :m
    @test target_index(t) == 3

    # The target_index is a value, not a type parameter: every position shares one type.
    @test typeof(IndexedTarget(:m, 1)) === typeof(IndexedTarget(:m, 2))
end
