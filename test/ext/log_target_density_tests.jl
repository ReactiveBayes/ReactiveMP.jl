@testitem "LogTargetDensity univariate" begin
    using LogDensityProblems, ReactiveMP
   
    ext = Base.get_extension(ReactiveMP, :ReactiveMPProjectionExt)

    @test !isnothing(ext)

    using .ext

    μ = x -> -x^2
    ltd = LogTargetDensity(μ)
    @test ltd.dim == ()
    @test LogDensityProblems.dimension(ltd) == 1
    for x in -3:0.1:3
        @test LogDensityProblems.logdensity(ltd, x) == μ(x)
    end
   
end


@testitem "LogTargetDensity Multivariate" begin
    using LogDensityProblems, ReactiveMP, ExponentialFamily, Distributions
   
    ext = Base.get_extension(ReactiveMP, :ReactiveMPProjectionExt)

    @test !isnothing(ext)

    using .ext

    μ = x -> -x'*x
    ltd = LogTargetDensity(Multivariate, (3, ), μ)
    @test ltd.dim == (3, )
    @test LogDensityProblems.dimension(ltd) == 3
    for x in ([1,2,3], ones(3), randn(3))
        @test LogDensityProblems.logdensity(ltd, x) == μ(x)
    end
   
end



@testitem "LogTargetDensity Matrixvariate" begin
    using LogDensityProblems, ReactiveMP, ExponentialFamily, Distributions, LinearAlgebra
   
    ext = Base.get_extension(ReactiveMP, :ReactiveMPProjectionExt)

    @test !isnothing(ext)

    using .ext

    μ = x -> -tr(x'*x)
    ltd = LogTargetDensity(Multivariate, (3, 2), μ)
    @test ltd.dim == (3, 2)
    @test LogDensityProblems.dimension(ltd) == 6
    for x in ([1 2; 1 3; 4 2], ones(3,2), randn(3,2))
        @test LogDensityProblems.logdensity(ltd, x) == μ(x)
    end
   
end