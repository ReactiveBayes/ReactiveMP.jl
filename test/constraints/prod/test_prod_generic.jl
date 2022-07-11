module ReactiveMPProdGenericTest

using Test
using ReactiveMP
using Random
using LinearAlgebra
using Distributions

import ReactiveMP: getdist, prod

@testset "ProdGeneric" begin
    × = (x, y) -> prod(ProdGeneric(), x, y)

    @testset "ProdGeneric should use ProdAnalytical where possible" begin
        struct DummyDistribution end

        Distributions.support(::DummyDistribution) = (-Inf, Inf)

        ReactiveMP.prod_analytical_rule(::Type{DummyDistribution}, ::Type{DummyDistribution}) = ProdAnalyticalRuleAvailable()
        ReactiveMP.prod(::ProdAnalytical, ::DummyDistribution, ::DummyDistribution)           = DummyDistribution()

        @test prod(ProdGeneric(), DummyDistribution(), DummyDistribution()) === DummyDistribution()
    end

    @testset "ProdGeneric should simplify a product tree if closed form product available for leafes" begin
        struct DummyDistribution11 end
        struct DummyDistribution12 end
        struct DummyDistribution13 end
        struct DummyDistribution14 end

        Distributions.support(::DummyDistribution11) = (-Inf, Inf)
        Distributions.support(::DummyDistribution12) = (-Inf, Inf)
        Distributions.support(::DummyDistribution13) = (-Inf, Inf)
        Distributions.support(::DummyDistribution14) = (-Inf, Inf)

        ReactiveMP.prod_analytical_rule(::Type{DummyDistribution12}, ::Type{DummyDistribution13}) = ProdAnalyticalRuleAvailable()
        ReactiveMP.prod_analytical_rule(::Type{DummyDistribution13}, ::Type{DummyDistribution12}) = ProdAnalyticalRuleAvailable()
        ReactiveMP.prod(::ProdAnalytical, ::DummyDistribution12, ::DummyDistribution13)           = DummyDistribution14()
        ReactiveMP.prod(::ProdAnalytical, ::DummyDistribution13, ::DummyDistribution12)           = DummyDistribution14()

        d1 = DummyDistribution11()
        d2 = DummyDistribution12()
        d3 = DummyDistribution13()
        d4 = DummyDistribution14()

        @test (d1 × d2) × d3 === d1 × d4
        @test (d2 × d1) × d3 === d4 × d1

        @test d3 × (d2 × d1) === d4 × d1
        @test d3 × (d1 × d2) === d1 × d4

        @test (d2 × d2) × (d3 × d3) === (d4 × d4)
        @test (d3 × d3) × (d2 × d2) === (d4 × d4)
    end

    @testset "ProdGeneric should create a product tree if closed form product is not available" begin
        struct DummyDistribution21 end
        struct DummyDistribution22 end
        struct DummyDistribution23 end

        Distributions.support(::DummyDistribution21) = (-Inf, Inf)
        Distributions.support(::DummyDistribution22) = (-Inf, Inf)
        Distributions.support(::DummyDistribution23) = (-Inf, Inf)

        d1 = DummyDistribution21()
        d2 = DummyDistribution22()
        d3 = DummyDistribution23()

        @test d1 × d2 === DistProduct(DummyDistribution21(), DummyDistribution22())
        @test (d1 × d2) × d3 ===
              DistProduct(DistProduct(DummyDistribution21(), DummyDistribution22()), DummyDistribution23())
    end

    @testset "ProdGeneric should create a linearised product tree if closed form product is not available, but objects are of the same type" begin
        struct DummyDistribution31 end
        struct DummyDistribution32 end

        Distributions.support(::DummyDistribution31) = (-Inf, Inf)
        Distributions.support(::DummyDistribution32) = (-Inf, Inf)

        d1 = DummyDistribution31()
        d2 = DummyDistribution32()

        @test d1 × d2 === DistProduct(DummyDistribution31(), DummyDistribution32())
        @test d1 × d2 × d2 × d2 isa
              DistProduct{DummyDistribution31, GenericLogPdfVectorisedProduct{DummyDistribution32}}
        @test (d1 × d2 × d2 × d2) × d1 × d1 isa DistProduct{
            GenericLogPdfVectorisedProduct{DummyDistribution31},
            GenericLogPdfVectorisedProduct{DummyDistribution32}
        }
    end
end

end
