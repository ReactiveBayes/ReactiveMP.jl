@testitem "targets" tags = [:base] begin
    using MessagePassingRulesBase: Target, IndexedTarget, edge, index

    @test Target(:out) === Target{:out}()
    @test edge(Target(:out)) === :out

    t = IndexedTarget(:m, 3)
    @test t isa IndexedTarget{:m}
    @test edge(t) === :m
    @test index(t) == 3

    # The index is a value, not a type parameter: every position shares one type.
    @test typeof(IndexedTarget(:m, 1)) === typeof(IndexedTarget(:m, 2))
end
