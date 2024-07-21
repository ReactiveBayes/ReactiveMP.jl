@testitem "LogTargetDensity" begin
    using LogDensityProblems, ReactiveMP
   
    ext = Base.get_extension(ReactiveMP, :ReactiveMPProjectionExt)

    @test !isnothing(ext)

    using .ext

    μ = x -> -x^2
    ltd = LogTargetDensity(3, μ)
    @test ltd.dim == 3
    @test LogDensityProblems.dimension(ltd) == 3
    for x in -3:0.1:3
        @test LogDensityProblems.logdensity(ltd, x) == μ(x)
    end
   
end




