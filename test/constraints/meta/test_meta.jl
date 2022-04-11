module ReactiveMPMetaSpecificationHelpers

using Test
using ReactiveMP
using GraphPPL
using Distributions

import ReactiveMP: resolve_meta

@testset "Meta specification" begin

    struct SomeNode end
    struct SomeOtherNode end

    ReactiveMP.as_node_symbol(::Type{ <: SomeNode }) = :SomeNode
    ReactiveMP.as_node_symbol(::Type{ <: SomeOtherNode }) = :SomeOtherNode

    @testset "Use case #1" begin 

        meta = @meta begin 
            SomeNode(x, y) -> "meta"
        end

        model = FactorGraphModel()

        x = randomvar(model, :x)
        y = randomvar(model, :y)
        z = randomvar(model, :z)

        @test resolve_meta(meta, model, SomeOtherNode, (x, y)) === nothing
        @test resolve_meta(meta, model, SomeNode, (x, y)) == "meta"
        @test resolve_meta(meta, model, SomeNode, (y, x)) == "meta"
        @test resolve_meta(meta, model, SomeNode, (x, y, z)) == "meta"
        @test resolve_meta(meta, model, SomeNode, (x, y, x, y)) == "meta"
        @test resolve_meta(meta, model, SomeNode, (x, y, x, y, z, z)) == "meta"
        @test resolve_meta(meta, model, SomeNode, (y, x, z)) == "meta"
        @test resolve_meta(meta, model, SomeNode, (y, z, x)) == "meta"
        @test resolve_meta(meta, model, SomeNode, (x, )) === nothing
        @test resolve_meta(meta, model, SomeNode, (x, z)) === nothing
        @test resolve_meta(meta, model, SomeNode, (y, )) === nothing
        @test resolve_meta(meta, model, SomeNode, (y, z)) === nothing
        @test resolve_meta(meta, model, SomeNode, (z, )) === nothing

    end

    @testset "Use case #2" begin 

        @meta function makemeta(flag)
            if flag
                SomeNode(x, y) -> "meta1"
            else
                SomeNode(x, y) -> "meta2"
            end
        end

        model = FactorGraphModel()

        x = randomvar(model, :x)
        y = randomvar(model, :y)
        z = randomvar(model, :z)

        for (meta, result) in ((makemeta(true), "meta1"), (makemeta(false), "meta2"))
            @test resolve_meta(meta, model, SomeOtherNode, (x, y)) === nothing
            @test resolve_meta(meta, model, SomeNode, (x, y)) == result
            @test resolve_meta(meta, model, SomeNode, (y, x)) ==  result
            @test resolve_meta(meta, model, SomeNode, (x, y, z)) ==  result
            @test resolve_meta(meta, model, SomeNode, (x, y, x, y)) ==  result
            @test resolve_meta(meta, model, SomeNode, (x, y, x, y, z, z)) ==  result
            @test resolve_meta(meta, model, SomeNode, (y, x, z)) ==  result
            @test resolve_meta(meta, model, SomeNode, (y, z, x)) ==  result
            @test resolve_meta(meta, model, SomeNode, (x, )) === nothing
            @test resolve_meta(meta, model, SomeNode, (x, z)) === nothing
            @test resolve_meta(meta, model, SomeNode, (y, )) === nothing
            @test resolve_meta(meta, model, SomeNode, (y, z)) === nothing
            @test resolve_meta(meta, model, SomeNode, (z, )) === nothing
        end

    end

    @testset "Use case #3" begin 

        meta = @meta begin 
            SomeNode(x, y) -> "meta1"
            SomeNode(z, y) -> "meta2"
        end

        model = FactorGraphModel()

        x = randomvar(model, :x, 10)
        y = randomvar(model, :y, 10)
        z = randomvar(model, :z, 10)

        @test resolve_meta(meta, model, SomeNode, (x[1], z[1])) === nothing
        @test resolve_meta(meta, model, SomeNode, (x[1], z[1], z[2])) === nothing
        @test resolve_meta(meta, model, SomeNode, (x[1], x[2], z[1])) === nothing
        @test resolve_meta(meta, model, SomeNode, (x[1], y[1])) == "meta1"
        @test resolve_meta(meta, model, SomeNode, (x[1], x[2], y[1])) == "meta1"
        @test resolve_meta(meta, model, SomeNode, (y[1], y[2], x[1])) == "meta1"
        @test resolve_meta(meta, model, SomeNode, (z[1], z[2], y[1])) == "meta2"
        @test resolve_meta(meta, model, SomeNode, (y[1], y[2], z[1])) == "meta2"
        @test_throws ErrorException resolve_meta(meta, model, SomeNode, (x[1], y[1], z[1])) 
        @test_throws ErrorException resolve_meta(meta, model, SomeNode, (z[1], y[1], x[1])) 

    end

    @testset "Use case #4" begin 

        meta = @meta begin 
            SomeNode(x, y) -> "meta1"
        end

        model = FactorGraphModel()

        x = randomvar(model, :x, 10)
        y = randomvar(model, :y)
        tmp = randomvar(model, ReactiveMP.randomvar_options_set_proxy_variables((y, )), :tmp)

        ReactiveMP.setanonymous!(tmp, true)

        @test resolve_meta(meta, model, SomeNode, (x[1], tmp)) == "meta1"
        @test resolve_meta(meta, model, SomeNode, (x[1], x[2], tmp)) == "meta1"
        @test resolve_meta(meta, model, SomeNode, (tmp, x[1])) == "meta1"
        @test resolve_meta(meta, model, SomeNode, (tmp, x[1], x[2])) == "meta1"

        @test resolve_meta(meta, model, SomeNode, (y, )) === nothing
        @test resolve_meta(meta, model, SomeNode, (tmp, )) === nothing
        for i in 1:10
            @test resolve_meta(meta, model, SomeNode, (x[i], )) === nothing
            @test resolve_meta(meta, model, SomeOtherNode, (x[i], y)) === nothing
        end

    end

    @testset "Use case #5" begin 

        # Just many variables and statements
        meta = @meta begin
            HGF(x) -> 123 
            AR(x, y, z) -> 1
            AR(x1, y, z) -> 2
            AR(x2, y, z) -> 3
            AR(x3, y, z) -> 4
            AR(x4, y, z) -> 5
            AR(x5, y, z) -> 6
            AR(x6, y, z) -> 7
            AR(x7, y, z) -> 8
            AR(x8, y, z) -> 9
            AR(x9, y, z) -> 10
            AR(x10, y, z) -> 11
            AR(x1, y1, z) -> 12
            AR(x2, y2, z) -> 13
            AR(x3, y3, z) -> 14
            AR(x4, y4, z) -> 15
            AR(x5, y5, z) -> 16
            AR(x6, y6, z3) -> 17
            AR(x7, y7, z4) -> 18
            AR(x8, y8, z) -> 19
            AR(x9, y9, z) -> 20
            AR(x10, y10, z) -> 21
        end

        model = FactorGraphModel()

        z = randomvar(model, :z)
        x = randomvar(model, :x10)
        y = randomvar(model, :y10)
    
        @test ReactiveMP.resolve_meta(meta, model, AR, (z, x, y)) === 21
        @test ReactiveMP.resolve_meta(meta, model, AR, (x, z, y)) === 21
        @test ReactiveMP.resolve_meta(meta, model, AR, (x, y, z)) === 21

    end
    
    # Errors

    @testset "Error case #1" begin 

        meta = @meta begin 
            SomeNode(x, y) -> "meta"
            SomeNode(y, x) -> "meta"
        end

        model = FactorGraphModel()

        x = randomvar(model, :x)
        y = randomvar(model, :y)
        z = randomvar(model, :z)

        @test resolve_meta(meta, model, SomeOtherNode, (x, y)) === nothing
        @test_throws ErrorException resolve_meta(meta, model, SomeNode, (x, y))
        @test_throws ErrorException resolve_meta(meta, model, SomeNode, (y, x))
        @test_throws ErrorException resolve_meta(meta, model, SomeNode, (x, y, z))
        @test_throws ErrorException resolve_meta(meta, model, SomeNode, (x, y, x, y))
        @test_throws ErrorException resolve_meta(meta, model, SomeNode, (x, y, x, y, z, z))
        @test_throws ErrorException resolve_meta(meta, model, SomeNode, (y, x, z))
        @test_throws ErrorException resolve_meta(meta, model, SomeNode, (y, z, x))
        @test resolve_meta(meta, model, SomeNode, (x, )) === nothing
        @test resolve_meta(meta, model, SomeNode, (x, z)) === nothing
        @test resolve_meta(meta, model, SomeNode, (y, )) === nothing
        @test resolve_meta(meta, model, SomeNode, (y, z)) === nothing
        @test resolve_meta(meta, model, SomeNode, (z, )) === nothing

    end

end

end