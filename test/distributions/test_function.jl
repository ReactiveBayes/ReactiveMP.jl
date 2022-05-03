module AbstractContinuousGenericLogPdfTest

using Test
using ReactiveMP
using Distributions
using Random

import ReactiveMP: getdomain, AbstractContinuousGenericLogPdf

import DomainIntegrals
import DomainSets

@testset "Generic Functional Distributions" begin

    ## Utility methods for tests 

    # These methods are inaccurate and relies on various approximation methods, which may fail in different scenarios
    # This should not be used though anywhere in the real code, but only in tests
    # Current implementation of `isapprox` method supports only FullSpace and HalfLine domains with limited accuracy
    function Base.isapprox(left::AbstractContinuousGenericLogPdf, right::AbstractContinuousGenericLogPdf; kwargs...)
        if (getdomain(left) !== getdomain(right)) || (value_support(typeof(left)) !== value_support(typeof(right))) || (variate_form(typeof(left)) !== variate_form(typeof(right)))
            return false
        end
        return culogpdf__isapprox(getdomain(left), left, right; kwargs...)
    end

    # https://en.wikipedia.org/wiki/Gauss–Hermite_quadrature
    function culogpdf__isapprox(domain::DomainSets.FullSpace, left::AbstractContinuousGenericLogPdf, right::AbstractContinuousGenericLogPdf; kwargs...)
        return isapprox(zero(eltype(domain)), DomainIntegrals.integral(DomainIntegrals.Q_GaussHermite(32), (x) -> exp(x ^ 2) * abs(left(x) - right(x))); kwargs...)
    end

    # https://en.wikipedia.org/wiki/Gauss–Laguerre_quadrature
    function culogpdf__isapprox(domain::DomainSets.HalfLine, left::AbstractContinuousGenericLogPdf, right::AbstractContinuousGenericLogPdf; kwargs...)
        return isapprox(zero(eltype(domain)), DomainIntegrals.integral(DomainIntegrals.Q_GaussLaguerre(32), (x) -> exp(x) * abs(left(x) - right(x))); kwargs...)
    end

    # We do not check typeof of a different functions because in most of the cases lambdas have different types, but they can still be the same
    function is_typeof_equal(::ContinuousUnivariateLogPdf{D, F1}, ::ContinuousUnivariateLogPdf{D, F2}) where { D, F1 <: Function, F2 <: Function }
        return true
    end

    @testset "ContinuousUnivariateLogPdf" begin
        
        @testset "Constructor" begin 
            f  = (x) -> -x ^ 2
            d1 = ContinuousUnivariateLogPdf(f)
            d2 = ContinuousUnivariateLogPdf(DomainSets.FullSpace(), f)

            @test typeof(d1) === typeof(d2)
            @test d1 ≈ d2

            @test_throws AssertionError ContinuousUnivariateLogPdf(DomainSets.FullSpace() ^ 2, f)
        end

        @testset "Intentional errors" begin
            dist = ContinuousUnivariateLogPdf((x) -> x)
            @test_throws ErrorException mean(dist)
            @test_throws ErrorException median(dist)
            @test_throws ErrorException mode(dist)
            @test_throws ErrorException var(dist)
            @test_throws ErrorException std(dist)
            @test_throws ErrorException cov(dist)
            @test_throws ErrorException invcov(dist)
            @test_throws ErrorException entropy(dist)
            @test_throws ErrorException precision(dist)
        end

        @testset "pdf/logpdf" begin 
            d1 = ContinuousUnivariateLogPdf(DomainSets.FullSpace(), (x) -> -x ^ 2)
            
            f32_points1 = range(Float32(-10.0), Float32(10.0), length = 50)
            f64_points1 = range(-10.0, 10.0, length = 50)
            bf_points1  = range(BigFloat(-10.0), BigFloat(10.0), length = 50)
            points1 = vcat(f32_points1, f64_points1, bf_points1)

            @test all(map(p -> -p^2 == d1(p), points1))
            @test all(map(p -> -p^2 == logpdf(d1, p), points1))
            @test all(map(p -> exp(-p^2) == pdf(d1, p), points1))
            @test all(map(p -> -p^2 == d1([ p ]), points1))
            @test all(map(p -> -p^2 == logpdf(d1, [ p ]), points1))
            @test all(map(p -> exp(-p^2) == pdf(d1, [ p ]), points1))

            d2 = ContinuousUnivariateLogPdf(DomainSets.HalfLine(), (x) -> -x ^ 4)
            
            f32_points2 = range(Float32(0.0), Float32(10.0), length = 50)
            f64_points2 = range(0.0, 10.0, length = 50)
            bf_points2  = range(BigFloat(0.0), BigFloat(10.0), length = 50)
            points2 = vcat(f32_points2, f64_points2, bf_points2)

            @test all(map(p -> -p^4 == d2(p), points2))
            @test all(map(p -> -p^4 == logpdf(d2, p), points2))
            @test all(map(p -> exp(-p^4) == pdf(d2, p), points2))
            @test all(map(p -> -p^4 == d2([ p ]), points2))
            @test all(map(p -> -p^4 == logpdf(d2, [ p ]), points2))
            @test all(map(p -> exp(-p^4) == pdf(d2, [ p ]), points2))

            @test_throws AssertionError d2(-1.0)
            @test_throws AssertionError logpdf(d2, -1.0)
            @test_throws AssertionError pdf(d2, -1.0)
            @test_throws AssertionError d2([ -1.0 ])
            @test_throws AssertionError logpdf(d2, [ -1.0 ])
            @test_throws AssertionError pdf(d2, [ -1.0 ])

            @test_throws AssertionError d2(Float32(-1.0))
            @test_throws AssertionError logpdf(d2, Float32(-1.0))
            @test_throws AssertionError pdf(d2, Float32(-1.0))
            @test_throws AssertionError d2([ Float32(-1.0) ])
            @test_throws AssertionError logpdf(d2, [ Float32(-1.0) ])
            @test_throws AssertionError pdf(d2, [ Float32(-1.0) ])

            @test_throws AssertionError d2(BigFloat(-1.0))
            @test_throws AssertionError logpdf(d2, BigFloat(-1.0))
            @test_throws AssertionError pdf(d2, BigFloat(-1.0))
            @test_throws AssertionError d2([ BigFloat(-1.0) ])
            @test_throws AssertionError logpdf(d2, [ BigFloat(-1.0) ])
            @test_throws AssertionError pdf(d2, [ BigFloat(-1.0) ])

            d3 = ContinuousUnivariateLogPdf(DomainSets.FullSpace(Float32), (x) -> -x ^ 2)

            @test all(map(p -> -p^2 == d3(p), points1))
            @test all(map(p -> -p^2 == logpdf(d3, p), points1))
            @test all(map(p -> exp(-p^2) == pdf(d3, p), points1))
            @test all(map(p -> -p^2 == d3([ p ]), points1))
            @test all(map(p -> -p^2 == logpdf(d3, [ p ]), points1))
            @test all(map(p -> exp(-p^2) == pdf(d3, [ p ]), points1))

            d4 = ContinuousUnivariateLogPdf(DomainSets.FullSpace(BigFloat), (x) -> -x ^ 2)

            @test all(map(p -> -p^2 == d4(p), points1))
            @test all(map(p -> -p^2 == logpdf(d4, p), points1))
            @test all(map(p -> exp(-p^2) == pdf(d4, p), points1))
            @test all(map(p -> -p^2 == d4([ p ]), points1))
            @test all(map(p -> -p^2 == logpdf(d4, [ p ]), points1))
            @test all(map(p -> exp(-p^2) == pdf(d4, [ p ]), points1))

            d5 = ContinuousUnivariateLogPdf(DomainSets.HalfLine{Float32}(), (x) -> -x ^ 2)

            @test all(map(p -> -p^2 == d5(p), points2))
            @test all(map(p -> -p^2 == logpdf(d5, p), points2))
            @test all(map(p -> exp(-p^2) == pdf(d5, p), points2))
            @test all(map(p -> -p^2 == d5([ p ]), points2))
            @test all(map(p -> -p^2 == logpdf(d5, [ p ]), points2))
            @test all(map(p -> exp(-p^2) == pdf(d5, [ p ]), points2))

            d6 = ContinuousUnivariateLogPdf(DomainSets.HalfLine{BigFloat}(), (x) -> -x ^ 2)

            @test all(map(p -> -p^2 == d6(p), points2))
            @test all(map(p -> -p^2 == logpdf(d6, p), points2))
            @test all(map(p -> exp(-p^2) == pdf(d6, p), points2))
            @test all(map(p -> -p^2 == d6([ p ]), points2))
            @test all(map(p -> -p^2 == logpdf(d6, [ p ]), points2))
            @test all(map(p -> exp(-p^2) == pdf(d6, [ p ]), points2))
        end

        @testset "support" begin 
            d1 = ContinuousUnivariateLogPdf(DomainSets.FullSpace(), (x) -> 1.0)
            @test minimum(support(d1)) === -Inf
            @test maximum(support(d1)) === Inf

            d2 = ContinuousUnivariateLogPdf(DomainSets.HalfLine(), (x) -> 1.0)
            @test minimum(support(d2)) === 0.0
            @test maximum(support(d2)) === Inf
        end

        @testset "vague" begin
            d = vague(ContinuousUnivariateLogPdf)

            @test typeof(d) <: ContinuousUnivariateLogPdf
            @test d ≈ ContinuousUnivariateLogPdf(DomainSets.FullSpace(), (x) -> 1.0)
        end

        @testset "prod" begin
            d1 = ContinuousUnivariateLogPdf(DomainSets.FullSpace(), (x) -> 2.0 * -x ^ 2)
            d2 = ContinuousUnivariateLogPdf(DomainSets.FullSpace(), (x) -> 3.0 * -x ^ 2)

            pr1 = prod(ProdAnalytical(), d1, d2)
            pt1 = ContinuousUnivariateLogPdf(DomainSets.FullSpace(), (x) -> logpdf(d1, x) + logpdf(d2, x))

            @test getdomain(pr1) === getdomain(d1)
            @test getdomain(pr1) === getdomain(d2)
            @test variate_form(typeof(pr1)) === variate_form(typeof(d1))
            @test variate_form(typeof(pr1)) === variate_form(typeof(d2))
            @test value_support(typeof(pr1)) === value_support(typeof(d1))
            @test value_support(typeof(pr1)) === value_support(typeof(d2))
            @test support(pr1) === support(d1)
            @test support(pr1) === support(d2)
            @test isapprox(pr1, pt1, atol = 1e-12)

            d3 = ContinuousUnivariateLogPdf(DomainSets.HalfLine(), (x) -> 2.0 * -x ^ 2)
            d4 = ContinuousUnivariateLogPdf(DomainSets.HalfLine(), (x) -> 3.0 * -x ^ 2)

            pr2 = prod(ProdAnalytical(), d3, d4)
            pt2 = ContinuousUnivariateLogPdf(DomainSets.HalfLine(), (x) -> logpdf(d3, x) + logpdf(d4, x))

            @test getdomain(pr2) === getdomain(d3)
            @test getdomain(pr2) === getdomain(d4)
            @test variate_form(typeof(pr2)) === variate_form(typeof(d3))
            @test variate_form(typeof(pr2)) === variate_form(typeof(d4))
            @test value_support(typeof(pr2)) === value_support(typeof(d3))
            @test value_support(typeof(pr2)) === value_support(typeof(d4))
            @test support(pr2) === support(d3)
            @test support(pr2) === support(d4)
            @test isapprox(pr2, pt2, atol = 1e-12)

            @test !isapprox(pr1, pr2, atol = 1e-12)

            d5 = ContinuousUnivariateLogPdf(DomainSets.FullSpace(), (x) -> 2.0 * -x ^ 2)
            d6 = ContinuousUnivariateLogPdf(DomainSets.HalfLine(), (x) -> 2.0 * -x ^ 2)
            @test_throws AssertionError prod(ProdAnalytical(), d5, d6)
        end

        @testset "vectorised-prod" begin
            f = (x) -> 2.0 * -x ^ 2
            d1 = ContinuousUnivariateLogPdf(DomainSets.FullSpace(), f)
            d2 = ContinuousUnivariateLogPdf(DomainSets.FullSpace(), f)
            d3 = ContinuousUnivariateLogPdf(DomainSets.FullSpace(), (x) -> f(x) + f(x))

            pr1 = prod(ProdAnalytical(), d1, d2)

            @test pr1 isa ContinuousGenericLogPdfVectorisedProduct
            @test getdomain(pr1) === getdomain(d1)
            @test getdomain(pr1) === getdomain(d2)
            @test variate_form(typeof(pr1)) === variate_form(typeof(d1))
            @test variate_form(typeof(pr1)) === variate_form(typeof(d2))
            @test value_support(typeof(pr1)) === value_support(typeof(d1))
            @test value_support(typeof(pr1)) === value_support(typeof(d2))
            @test support(pr1) === support(d1)
            @test support(pr1) === support(d2)
            @test isapprox(pr1, d3, atol = 1e-12)
        end

        @testset "convert" begin
            d = DomainSets.FullSpace()
            l = (x) -> 1.0

            c = convert(ContinuousUnivariateLogPdf, d, l)
            @test typeof(c) <: ContinuousUnivariateLogPdf
            @test isapprox(c, ContinuousUnivariateLogPdf(d, l), atol = 1e-12)

            c2 = convert(ContinuousUnivariateLogPdf, c)
            @test typeof(c2) <: ContinuousUnivariateLogPdf
            @test isapprox(c2, ContinuousUnivariateLogPdf(d, l), atol = 1e-12)

        end

    end

end

end
