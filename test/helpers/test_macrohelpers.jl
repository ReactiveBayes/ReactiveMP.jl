module ReactiveMPMacroHelpersTest

using Test
using ReactiveMP
using MacroTools

import ReactiveMP.MacroHelpers: bottom_type, upper_type, strip_type_parameters

@testset "Macro helpers" begin
    @testset "bottom_type" begin
        @test bottom_type(:(Int)) === :Int
        @test bottom_type(:(Type{Int})) === :Int
        @test bottom_type(:(Type{Vector{Int}})) == :(Vector{Int})
        @test bottom_type(:(Type{<:Real})) === :Real
        @test bottom_type(:(Type{<:Vector{Int}})) == :(Vector{Int})
        @test bottom_type(:(typeof(+))) === :+
    end

    @testset "upper_type" begin
        @test upper_type(:(Int)) == :(Type{<:Int})
        @test upper_type(:(Type{Int})) == :(Type{<:Int})
        @test upper_type(:(Type{Vector{Int}})) == :(Type{<:Vector{Int}})
        @test upper_type(:(Type{<:Real})) == :(Type{<:Real})
        @test upper_type(:(Type{<:Vector{Int}})) == :(Type{<:Vector{Int}})
        @test upper_type(:(typeof(+))) == :(typeof(+))
    end

    @testset "strip_type_parameters" begin
        @test strip_type_parameters(:(Int)) === :Int
        @test strip_type_parameters(:(Type{Int})) === :Type
        @test strip_type_parameters(:(Vector{Int})) === :Vector
        @test strip_type_parameters(:(Type{<:Real})) === :Type
        @test strip_type_parameters(:(Vector{<:Real})) === :Vector
        @test strip_type_parameters(:(typeof(+))) == :(typeof(+))
    end
end

end
