@testitem "quality:aqua" tags = [:quality] begin
    using Aqua, FlowMessagePassingRules
    Aqua.test_all(FlowMessagePassingRules)
end

@testitem "quality:no ambiguities" tags = [:quality] begin
    using FlowMessagePassingRules, Test
    @test isempty(Test.detect_ambiguities(FlowMessagePassingRules; recursive = true))
end
