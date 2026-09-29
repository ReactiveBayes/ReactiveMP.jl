@testitem "helpers:diageye" tags = [:helpers] begin
    using StandardMessagePassingRules, MessagePassingRulesBase

    # Exported for models, as MessagePassingRulesBase's, with its element type or without one.
    @test diageye === MessagePassingRulesBase.diageye
    @test diageye(Float32, 2) == Float32[1 0; 0 1] && eltype(diageye(Float32, 2)) === Float32
    @test diageye(3) == [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0] && eltype(diageye(3)) === Float64
end
