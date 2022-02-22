module ReactiveMPFactorisationSpecTest 

using Test
using ReactiveMP 

import ReactiveMP: FunctionalIndex
import ReactiveMP: CombinedRange, SplittedRange, is_splitted
import ReactiveMP: __as_unit_range, __factorisation_specification_resolve_index

@testset "FactorisationSpec" begin 

    @testset "CombinedRange" begin 
        for left in 1:3, right in 10:13
            cr = CombinedRange(left, right)

            @test firstindex(cr) === left
            @test lastindex(cr)  === right
            @test !is_splitted(cr)
            
            for i in left:right
                @test i ∈ cr
                @test !((i + lastindex(cr) + 1) ∈ cr)
            end
        end
    end

    @testset "SplittedRange" begin 
        for left in 1:3, right in 10:13
            cr = SplittedRange(left, right)

            @test firstindex(cr) === left
            @test lastindex(cr)  === right
            @test is_splitted(cr)
            
            for i in left:right
                @test i ∈ cr
                @test !((i + lastindex(cr) + 1) ∈ cr)
            end
        end
    end

    @testset "__as_unit_range" begin    
        for i in 1:3
            __as_unit_range(i) === i:i
        end

        for i in 1:3
            __as_unit_range(CombinedRange(i, i + 1)) === (i):(i + 1)
            __as_unit_range(SplittedRange(i, i + 1)) === (i):(i + 1)
        end
    end

    @testset "__factorisation_specification_resolve_index" begin
        collection = randomvar(:x, 3)
        
        @test __factorisation_specification_resolve_index(nothing, randomvar(:x)) === nothing
        @test_throws ErrorException __factorisation_specification_resolve_index(1, randomvar(:x))
        @test_throws ErrorException __factorisation_specification_resolve_index(FunctionalIndex{:begin}(firstindex), randomvar(:x))
        
        @test __factorisation_specification_resolve_index(nothing, collection) === nothing
        @test __factorisation_specification_resolve_index(1, collection) === 1
        @test __factorisation_specification_resolve_index(3, collection) === 3
        
        @test_throws ErrorException __factorisation_specification_resolve_index(6, collection)
        
        @test __factorisation_specification_resolve_index(FunctionalIndex{:begin}(firstindex) + 1, collection) === 2
        @test __factorisation_specification_resolve_index(FunctionalIndex{:begin}(firstindex) + 1 + 1, collection) === 3
        @test __factorisation_specification_resolve_index(FunctionalIndex{:end}(lastindex) - 1, collection) === 2
        
        @test_throws ErrorException __factorisation_specification_resolve_index(FunctionalIndex{:begin}(firstindex) + 100, collection)
        @test_throws ErrorException __factorisation_specification_resolve_index(FunctionalIndex{:end}(lastindex) - 100, collection)
        
        @test __factorisation_specification_resolve_index(CombinedRange(1, 3), collection) === CombinedRange(1, 3)
        @test __factorisation_specification_resolve_index(CombinedRange(1, 2), collection) === CombinedRange(1, 2)
        @test __factorisation_specification_resolve_index(CombinedRange(FunctionalIndex{:begin}(firstindex), 2), collection) === CombinedRange(1, 2)
        @test __factorisation_specification_resolve_index(CombinedRange(1, FunctionalIndex{:end}(lastindex)), collection) === CombinedRange(1, 3)
        @test __factorisation_specification_resolve_index(CombinedRange(FunctionalIndex{:begin}(firstindex) + 1, FunctionalIndex{:end}(lastindex) - 1), collection) === CombinedRange(2, 2)
        
        @test __factorisation_specification_resolve_index(SplittedRange(1, 3), collection) === SplittedRange(1, 3)
        @test __factorisation_specification_resolve_index(SplittedRange(1, 2), collection) === SplittedRange(1, 2)
        @test __factorisation_specification_resolve_index(SplittedRange(FunctionalIndex{:begin}(firstindex), 2), collection) === SplittedRange(1, 2)
        @test __factorisation_specification_resolve_index(SplittedRange(1, FunctionalIndex{:end}(lastindex)), collection) === SplittedRange(1, 3)
        @test __factorisation_specification_resolve_index(SplittedRange(FunctionalIndex{:begin}(firstindex) + 1, FunctionalIndex{:end}(lastindex) - 1), collection) === SplittedRange(2, 2)
        
        @test_throws ErrorException __factorisation_specification_resolve_index(CombinedRange(1, 5), collection)
        @test_throws ErrorException __factorisation_specification_resolve_index(SplittedRange(1, 5), collection)
        
        @test_throws ErrorException __factorisation_specification_resolve_index(CombinedRange(FunctionalIndex{:begin}(firstindex) + 100, 2), collection)
        @test_throws ErrorException __factorisation_specification_resolve_index(CombinedRange(1, FunctionalIndex{:end}(lastindex) - 100), collection)
        @test_throws ErrorException __factorisation_specification_resolve_index(CombinedRange(FunctionalIndex{:begin}(firstindex) + 1, FunctionalIndex{:end}(lastindex) - 100), collection)
        @test_throws ErrorException __factorisation_specification_resolve_index(CombinedRange(FunctionalIndex{:begin}(firstindex) + 100, FunctionalIndex{:end}(lastindex)), collection)
        
        @test_throws ErrorException __factorisation_specification_resolve_index(SplittedRange(FunctionalIndex{:begin}(firstindex) + 100, 2), collection)
        @test_throws ErrorException __factorisation_specification_resolve_index(SplittedRange(1, FunctionalIndex{:end}(lastindex) - 100), collection)
        @test_throws ErrorException __factorisation_specification_resolve_index(SplittedRange(FunctionalIndex{:begin}(firstindex) + 1, FunctionalIndex{:end}(lastindex) - 100), collection)
        @test_throws ErrorException __factorisation_specification_resolve_index(SplittedRange(FunctionalIndex{:begin}(firstindex) + 100, FunctionalIndex{:end}(lastindex)), collection)
    end

end

end