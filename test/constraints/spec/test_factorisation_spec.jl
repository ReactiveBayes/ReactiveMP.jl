module ReactiveMPFactorisationSpecTest 

using Test
using ReactiveMP 

import ReactiveMP: CombinedRange, SplittedRange, is_splitted
import ReactiveMP: __as_unit_range

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

end

end