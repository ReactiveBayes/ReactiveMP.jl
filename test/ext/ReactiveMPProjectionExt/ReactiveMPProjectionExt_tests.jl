@testitem "DivisionOf" begin
    using ExponentialFamily, ExponentialFamilyProjection, BayesBase

    ext = Base.get_extension(ReactiveMP, :ReactiveMPProjectionExt)

    @test !isnothing(ext)

    using .ext

    d1 = NormalMeanVariance(0, 1)
    d2 = NormalMeanVariance(1, 2)

    @test d1 ≈ prod(GenericProd(), ext.DivisionOf(d1, d2), d2)
    @test d1 ≈ prod(GenericProd(), d2, ext.DivisionOf(d1, d2))
end

@testitem "Extension availability with `ExponentialFamilyProjection`" begin
    using ExponentialFamilyProjection
    
    @test ReactiveMP.is_delta_node_compatible(CVIProjection()) === Val(true)
    
end