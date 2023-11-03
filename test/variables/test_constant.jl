module ReactiveMPConstVariableTest

using Test, ReactiveMP, Rocket, BayesBase, Distributions, ExponentialFamily

using LinearAlgebra: I

import ReactiveMP: collection_type, VariableIndividual, VariableVector, VariableArray, linear_index
import ReactiveMP: getconst, proxy_variables
import ReactiveMP: israndom, isproxy

@testset "ConstVariable" begin
    @testset "Simple creation" begin
        for sym in (:x, :y, :z), value in (1.0, 1.0, "asd", I, 0.3 * I, [1.0, 1.0], [1.0 0.0; 0.0 1.0], (x) -> 1)
            v = constvar(sym, value)

            @test !israndom(v)
            @test getconst(v) === value
            @test name(v) === sym
            @test collection_type(v) isa VariableIndividual
            @test proxy_variables(v) === nothing
            @test !isproxy(v)
        end

        for sym in (:x, :y, :z), n in (10, 20)
            vs = constvar(sym, (i) -> i + 1, n)

            @test !israndom(vs)
            @test length(vs) === n
            @test vs isa Vector
            @test all(v -> !israndom(v), vs)
            @test all(v -> name(v) === sym, vs)
            @test all(v -> collection_type(v) isa VariableVector, vs)
            @test all(t -> linear_index(collection_type(t[2])) === t[1], enumerate(vs))
            @test all(t -> getconst(t[2]) === t[1] + 1, enumerate(vs))
            @test !isproxy(vs)
            @test all(v -> !isproxy(v), vs)
        end

        for sym in (:x, :y, :z), l in (10, 20), r in (10, 20)
            for vs in (constvar(sym, (i) -> sum(i), l, r), constvar(sym, (i) -> sum(i), (l, r)))
                @test !israndom(vs)
                @test size(vs) === (l, r)
                @test length(vs) === l * r
                @test vs isa Matrix
                @test all(v -> !israndom(v), vs)
                @test all(v -> name(v) === sym, vs)
                @test all(v -> collection_type(v) isa VariableArray, vs)
                @test all(t -> linear_index(collection_type(t[2])) === t[1], enumerate(vs))
                @test all(i -> getconst(vs[i]) == sum(convert(Tuple, i)), CartesianIndices(axes(vs)))
                @test !isproxy(vs)
                @test all(v -> !isproxy(v), vs)
            end
        end
    end
end

end
