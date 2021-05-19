module DirichletTest

using Test
using ReactiveMP
using Distributions
using Random

@testset "Dirichlet" begin

    # Dirichlet comes from Distributions.jl and most of the things should be covered there
    # Here we test some extra ReactiveMP.jl specific functionality

    @testset "vague" begin
        @test_throws MethodError vague(Dirichlet)

        d1 = vague(Dirichlet, 2)

        @test typeof(d1) <: Dirichlet
        @test probvec(d1) == ones(2)

        d2 = vague(Dirichlet, 4)

        @test typeof(d2) <: Dirichlet
        @test probvec(d2) == ones(4)
    end

    @testset "prod" begin
        @test prod(ProdAnalytical(), Dirichlet([ 1.0, 1.0, 1.0 ]), Dirichlet([ 1.0, 1.0, 1.0 ])) == Dirichlet([ 1.0, 1.0, 1.0 ])
        @test prod(ProdAnalytical(), Dirichlet([ 1.1, 1.0, 2.0 ]), Dirichlet([ 1.0, 1.2, 1.0 ])) == Dirichlet([1.1, 1.2000000000000002, 2.0])
        @test prod(ProdAnalytical(), Dirichlet([ 1.1, 2.0, 2.0 ]), Dirichlet([ 3.0, 1.2, 5.0 ])) == Dirichlet([3.0999999999999996, 2.2, 6.0])
    end

    @testset "probvec" begin
        @test probvec(Dirichlet([ 1.0, 1.0, 1.0 ])) == [ 1.0, 1.0, 1.0 ]
        @test probvec(Dirichlet([ 1.1, 2.0, 2.0 ])) == [ 1.1, 2.0, 2.0 ]
        @test probvec(Dirichlet([ 3.0, 1.2, 5.0 ])) == [ 3.0, 1.2, 5.0 ]
    end

    @testset "logmean" begin
        @test logmean(Dirichlet([ 1.0, 1.0, 1.0 ])) ≈ [-1.5000000000000002, -1.5000000000000002, -1.5000000000000002]
        @test logmean(Dirichlet([ 1.1, 2.0, 2.0 ])) ≈ [-1.9517644694670657, -1.1052251939575213, -1.1052251939575213]
        @test logmean(Dirichlet([ 3.0, 1.2, 5.0 ])) ≈ [-1.2410879175727905, -2.4529121492634465, -0.657754584239457]
    end

    @testset "promote_variate_type" begin
        @test_throws MethodError promote_variate_type(Univariate, Dirichlet) 

        @test promote_variate_type(Multivariate, Dirichlet)  === Dirichlet
        @test promote_variate_type(Matrixvariate, Dirichlet) === MatrixDirichlet

        @test promote_variate_type(Multivariate, MatrixDirichlet)  === Dirichlet
        @test promote_variate_type(Matrixvariate, MatrixDirichlet) === MatrixDirichlet
    end

end

end
