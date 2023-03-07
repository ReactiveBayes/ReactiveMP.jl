module ReactiveMPDataVariableTest

using Test
using ReactiveMP
using Rocket

import ReactiveMP: DataVariableCreationOptions, MessageObservable
import ReactiveMP: collection_type, VariableIndividual, VariableVector, VariableArray, linear_index
import ReactiveMP: getconst, proxy_variables
import ReactiveMP: israndom, isproxy, isused, isconnected, setmessagein!, allows_missings

@testset "DataVariable" begin
    @testset "Simple creation" begin
        randomize_update(::Type{Missing}, size)                            = fill(missing, size)
        randomize_update(::Type{T}, size) where {T <: Union{Int, Float64}} = rand(T, size)
        randomize_update(::Type{V}, size) where {V <: AbstractVector}      = map(_ -> rand(eltype(V), 1), CartesianIndices(size))

        function test_updates(vs, type, size)
            nupdates     = 3
            updates      = []
            subscription = subscribe!(getmarginals(vs), (update) -> begin
                update_data = ReactiveMP.getdata.(update)
                if all(element -> element isa type, update_data)
                    push!(updates, update_data)
                end
            end)
            for _ in 1:nupdates
                update = randomize_update(type, size)
                update!(vs, update)
                @test all(last(updates) .=== update)
            end
            @test length(updates) === nupdates
            unsubscribe!(subscription)
            # Check if we do not receive updates after unsubscription
            update!(vs, randomize_update(type, size))
            @test length(updates) === nupdates
            return true
        end

        for sym in (:x, :y, :z), T in (Float64, Int64, Vector{Float64}), allow_missings in (true, false)
            options = DataVariableCreationOptions(T, nothing, Val(allow_missings))
            variable = datavar(options, sym, T)

            @test !israndom(variable)
            @test eltype(variable) === T
            @test name(variable) === sym
            @test allows_missings(variable) === allow_missings
            @test collection_type(variable) isa VariableIndividual
            @test proxy_variables(variable) === nothing
            @test !isproxy(variable)
            @test !isused(variable)
            @test !isconnected(variable)

            setmessagein!(variable, 1, MessageObservable())

            @test isused(variable)
            @test isconnected(variable)

            # `100` could a valid index, but messages should be initialized in order, previous was `1`
            @test_throws ErrorException setmessagein!(variable, 100, MessageObservable())
        end

        for sym in (:x, :y, :z), T in (Float64, Int64, Vector{Float64}), n in (10, 20), allow_missings in (true, false)
            options = DataVariableCreationOptions(T, nothing, Val(allow_missings))
            variables = datavar(options, sym, T, n)

            @test !israndom(variables)
            @test length(variables) === n
            @test variables isa Vector
            @test all(v -> !israndom(v), variables)
            @test all(v -> name(v) === sym, variables)
            @test all(v -> allows_missings(v) === allow_missings, variables)
            @test all(v -> collection_type(v) isa VariableVector, variables)
            @test all(t -> linear_index(collection_type(t[2])) === t[1], enumerate(variables))
            @test all(v -> eltype(v) === T, variables)
            @test !isproxy(variables)
            @test all(v -> !isproxy(v), variables)
            @test all(v -> !isused(v), variables)
            @test all(v -> !isconnected(v), variables)
            @test test_updates(variables, T, (n,))

            foreach(v -> setmessagein!(v, 1, MessageObservable()), variables)

            @test all(v -> isused(v), variables)
            @test all(v -> isconnected(v), variables)
        end

        for sym in (:x, :y, :z), T in (Float64, Int64, Vector{Float64}), l in (10, 20), r in (10, 20)
            for variables in (datavar(sym, T, l, r), datavar(sym, T, (l, r)))
                @test !israndom(variables)
                @test size(variables) === (l, r)
                @test length(variables) === l * r
                @test variables isa Matrix
                @test all(v -> !israndom(v), variables)
                @test all(v -> name(v) === sym, variables)
                @test all(v -> collection_type(v) isa VariableArray, variables)
                @test all(t -> linear_index(collection_type(t[2])) === t[1], enumerate(variables))
                @test all(v -> eltype(v) === T, variables)
                @test !isproxy(variables)
                @test all(v -> !isproxy(v), variables)
                @test all(v -> !isused(v), variables)
                @test test_updates(variables, T, (l, r))

                foreach(v -> setmessagein!(v, 1, MessageObservable()), variables)

                @test all(v -> isused(v), variables)
                @test all(v -> isconnected(v), variables)
            end
        end
    end

    @testset "Error indexing" begin
        # This test may be removed if we implement this feature in the future
        @test_throws ErrorException datavar(:x, Float64)[1]
    end
end

end
