@testitem "quality:aqua" tags = [:quality] begin
    using Aqua, FlowMessagePassingRules
    Aqua.test_all(FlowMessagePassingRules)
end

@testitem "quality:no ambiguities" tags = [:quality] begin
    using FlowMessagePassingRules, Test
    # v6's PermutationMatrix had 119 method ambiguities in 6.5.0, 18 of them among its own.
    @test isempty(Test.detect_ambiguities(FlowMessagePassingRules; recursive = true))
end
