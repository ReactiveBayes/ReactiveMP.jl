module InverseWishartNodeTest

using Test
using ReactiveMP
using Random

import ReactiveMP: make_node

@testset "InverseWishartNode" begin
    @testset "Creation" begin
        node = make_node(GammaInverse)

        @test functionalform(node) === GammaInverse
        @test sdtype(node) === Stochastic()
        @test name.(interfaces(node)) === (:out, :α, :β)
        # XXX ?
        @test factorisation(node) === ((1, 2, 3),)
        @test localmarginalnames(node) === (:out_α_β,)
        # XXX ?
        @test metadata(node) === nothing

        # XXX ?
        node = make_node(InverseWishart, FactorNodeCreationOptions(nothing, 1, nothing))
        @test metadata(node) === 1
    end

    # TODO: Average Energy testset
end # testset
end # module
