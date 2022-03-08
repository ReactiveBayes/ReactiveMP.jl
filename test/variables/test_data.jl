module ReactiveMPDataVariableTest

using Test
using ReactiveMP
using Rocket

import ReactiveMP: collection_type, VariableIndividual, VariableVector, VariableArray, linear_index
import ReactiveMP: getconst, proxy_variables
import ReactiveMP: israndom, isproxy

@testset "DataVariable" begin

    @testset "Simple creation" begin 

        for sym in (:x, :y, :z), type in (Float64, Int64, String, Vector{Float64})
            v = datavar(sym, type)

            @test !israndom(v)
            @test eltype(v) === type
            @test name(v) === sym
            @test collection_type(v) isa VariableIndividual
            @test proxy_variables(v) === nothing
            @test !isproxy(v)
        end

        for sym in (:x, :y, :z), type in (Float64, Int64, String, Vector{Float64}), n in (10, 20)
            vs = datavar(sym, type, n)
            
            @test !israndom(vs) 
            @test length(vs) === n
            @test vs isa Vector
            @test all(v -> !israndom(v), vs)
            @test all(v -> name(v) === sym, vs)
            @test all(v -> collection_type(v) isa VariableVector, vs)
            @test all(t -> linear_index(collection_type(t[2])) === t[1], enumerate(vs))
            @test all(v -> eltype(v) === type, vs)
            @test !isproxy(vs)
            @test all(v -> !isproxy(v), vs)
        end

        for sym in (:x, :y, :z), type in (Float64, Int64, String, Vector{Float64}), l in (10, 20), r in (10, 20)
            for vs in (datavar(sym, type, l, r), datavar(sym, type, (l, r)))
                @test !israndom(vs) 
                @test size(vs) === (l, r)
                @test length(vs) === l * r
                @test vs isa Matrix
                @test all(v -> !israndom(v), vs)
                @test all(v -> name(v) === sym, vs)
                @test all(v -> collection_type(v) isa VariableArray, vs)
                @test all(t -> linear_index(collection_type(t[2])) === t[1], enumerate(vs))
                @test all(v -> eltype(v) === type, vs)
                @test !isproxy(vs)
                @test all(v -> !isproxy(v), vs)
            end
        end


    end

end

end