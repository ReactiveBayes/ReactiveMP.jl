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

@testitem "CVIProjection" begin
    using ExponentialFamily, ExponentialFamilyProjection, BayesBase
    ext = Base.get_extension(ReactiveMP, :ReactiveMPProjectionExt)

    @test !isnothing(ext)

    using .ext

    named_tuple_types_projection = (out = MvNormalMeanCovariance, in = (Beta, Gamma))
    projection_dimensions        = (out = (4,), in = ((), ()))
    cviprojectionessentials      = CVIProjectionEssentials(projection_dims = projection_dimensions, projection_types = named_tuple_types_projection, initial_samples = (0.11, 20.0))
    cviprojection                = CVIProjection(projection_essentials = cviprojectionessentials)
    projection_dims_keys         = keys(ext.getcviprojectiondims(cviprojection))
    projection_types_keys        = keys(ext.getcviprojectiontypes(cviprojection))

    initial_samples = ext.getcviinitialsamples(cviprojection)
    @test initial_samples == (0.11, 20.0)

    @test ext.getcviprojectionconditioners(cviprojection) === nothing
    @test projection_dims_keys == (:out, :in)
    @test projection_types_keys == (:out, :in)
    @test values(ext.getcviprojectiontypes(cviprojection)) == (MvNormalMeanCovariance, (Beta, Gamma))
    @test values(ext.getcviprojectiondims(cviprojection)) == ((4,), ((), ()))
    @test haskey(ext.getcviprojectiontypes(cviprojection), :out) == true
    @test haskey(ext.getcviprojectiondims(cviprojection), :out) == true
    @test haskey(ext.getcviprojectiondims(cviprojection), :x) == false
    @test haskey(ext.getcviprojectiontypes(cviprojection), :in) == true
end
