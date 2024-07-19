# More tests are available in the `test/ext/ReactiveMPProjectionExt`

@testitem "Extension availability with `ExponentialFamilyProjection`" begin
    using ExponentialFamilyProjection

    @test ReactiveMP.is_delta_node_compatible(CVIProjection()) === Val(true)
    @test ReactiveMP.check_delta_node_compatibility(Val(true), CVIProjection()) === nothing
end

@testitem "Extension should not be available without `ExponentialFamilyProjection`" begin
    @test_throws "CVI projection requires `using ExponentialFamilyProjection` in the current session." ReactiveMP.check_delta_node_compatibility(Val(false), CVIProjection())
end
