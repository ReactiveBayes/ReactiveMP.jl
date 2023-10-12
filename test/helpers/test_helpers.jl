module ReactiveMPTestingHelpers

using Test
using ReactiveMP

import ReactiveMP: SkipIndexIterator, skipindex
import ReactiveMP: CountingReal
import ReactiveMP: FunctionalIndex

@testset "Helpers" begin
    @testset "SkipIndexIterator" begin
        s = skipindex(1:3, 2)
        @test typeof(s) <: SkipIndexIterator
        @test collect(s) == [1, 3]
        @test collect(skipindex(s, 1)) == [3]
    end

    @testset "FunctionalIndex" begin
        for N in 1:5
            collection = ones(N)
            @test FunctionalIndex{:nothing}(firstindex)(collection) === firstindex(collection)
            @test FunctionalIndex{:nothing}(lastindex)(collection) === lastindex(collection)
            @test (FunctionalIndex{:nothing}(firstindex) + 1)(collection) === firstindex(collection) + 1
            @test (FunctionalIndex{:nothing}(lastindex) - 1)(collection) === lastindex(collection) - 1
            @test (FunctionalIndex{:nothing}(firstindex) + 1 - 2 + 3)(collection) === firstindex(collection) + 1 - 2 + 3
            @test (FunctionalIndex{:nothing}(lastindex) - 1 + 2 - 3)(collection) === lastindex(collection) - 1 + 2 - 3
        end

        @test repr(FunctionalIndex{:begin}(firstindex)) === "(begin)"
        @test repr(FunctionalIndex{:begin}(firstindex) + 1) === "((begin) + 1)"
        @test repr(FunctionalIndex{:begin}(firstindex) - 1) === "((begin) - 1)"
        @test repr(FunctionalIndex{:begin}(firstindex) - 1 + 1) === "(((begin) - 1) + 1)"

        @test repr(FunctionalIndex{:end}(lastindex)) === "(end)"
        @test repr(FunctionalIndex{:end}(lastindex) + 1) === "((end) + 1)"
        @test repr(FunctionalIndex{:end}(lastindex) - 1) === "((end) - 1)"
        @test repr(FunctionalIndex{:end}(lastindex) - 1 + 1) === "(((end) - 1) + 1)"

        @test isbitstype(typeof((FunctionalIndex{:begin}(firstindex) + 1)))
        @test isbitstype(typeof((FunctionalIndex{:begin}(firstindex) - 1)))
        @test isbitstype(typeof((FunctionalIndex{:begin}(firstindex) + 1 + 1)))
        @test isbitstype(typeof((FunctionalIndex{:begin}(firstindex) - 1 + 1)))
    end

end

end
