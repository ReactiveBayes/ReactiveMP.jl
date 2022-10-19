module ReactiveMPLinearizationApproximationTest

using Test
using ReactiveMP

import ReactiveMP: Linearization

@testset "Linearization approximation method" begin
    
    @testset "__linearization_starts_at tests" begin 
        import ReactiveMP: __linearization_starts_at

        # `@inferred` is need for type-stability
        @test @inferred(__linearization_starts_at(((), (), ()))) === (1, 2, 3)
        @test @inferred(__linearization_starts_at(((3, ), (), ()))) === (1, 4, 5)
        @test @inferred(__linearization_starts_at(((), (3, ), ()))) === (1, 2, 5)
        @test @inferred(__linearization_starts_at(((), (), (3, )))) === (1, 2, 3)
        @test @inferred(__linearization_starts_at(((1, ), (1, ), (1, )))) === (1, 2, 3)
        @test @inferred(__linearization_starts_at(((2, ), (1, ), (1, )))) === (1, 3, 4)
        @test @inferred(__linearization_starts_at(((3, ), (1, ), (1, )))) === (1, 4, 5)

    end

    @testset "__linearization_as_vec tests" begin 
        import ReactiveMP: __linearization_as_vec

        # `@inferred` is need for type-stability
        @test @inferred(__linearization_as_vec((1, 2, 3))) == [ 1, 2, 3 ]
        @test @inferred(__linearization_as_vec((3, 2, 1))) == [ 3, 2, 1 ]
        @test @inferred(__linearization_as_vec((3, [ 2. ], 1))) == [ 3.0, 2.0, 1.0 ]
        @test @inferred(__linearization_as_vec(([ 3.0, 2.0 ], [ 2. ], 1))) == [ 3.0, 2.0, 2.0, 1.0 ]
        @test @inferred(__linearization_as_vec((3, 2, [ 1.0 0.0; 0.0 1.0 ]))) == [ 3.0, 2.0, 1.0, 0.0, 0.0, 1.0 ]
        @test @inferred(__linearization_as_vec(([ 3.0, 2.0 ], 2, [ 1.0 0.0; 0.0 1.0 ]))) == [ 3.0, 2.0, 2.0, 1.0, 0.0, 0.0, 1.0 ]

    end

    @testset "__linearization_splitjoin tests" begin 
        import ReactiveMP: __linearization_splitjoin

        # `@inferred` is need for type-stability
        @test @inferred(__linearization_splitjoin([ 1, 2, 3 ], ((), (), ()))) == (1, 2, 3)
        @test @inferred(__linearization_splitjoin([ 1, 2, 3 ], ((), (1, ), ()))) == (1, [ 2, ], 3)
        @test @inferred(__linearization_splitjoin([ 1, 2, 3 ], ((1, ), (1, ), ()))) == ([ 1, ], [ 2, ], 3)
        @test @inferred(__linearization_splitjoin([ 1, 2, 3 ], ((1, ), (1, ), (1, )))) == ([ 1, ], [ 2, ], [ 3, ])
        @test @inferred(__linearization_splitjoin([ 1, 2, 1, 0, 0, 1 ], ((1, ), (1, ), (2, 2)))) == ([ 1, ], [ 2, ], [ 1 0; 0 1 ])
        @test @inferred(__linearization_splitjoin([ 1, 0, 0, 1, 2, 1, 0, 0, 1 ], ((2, 2), (1, ), (2, 2)))) == ([ 1 0; 0 1 ], [ 2, ], [ 1 0; 0 1 ])
        @test @inferred(__linearization_splitjoin([ 1, 0, 0, 1, 2, 1, 0, 0, 1 ], ((2, 2), (), (2, 2)))) == ([ 1 0; 0 1 ], 2, [ 1 0; 0 1 ])

    end

    @testset "localLinearizationMultiIn tests" begin 
        import ReactiveMP: localLinearizationMultiIn

        @inferred localLinearizationMultiIn((x, y) -> x + y, (1, 2))
        @inferred localLinearizationMultiIn((x, y) -> x - y, (1, 2))
        @inferred localLinearizationMultiIn((x, y) -> x .- y, ([ 1.0, 2.0 ], 1.0))
        @inferred localLinearizationMultiIn((x, y) -> x .- y, ([ 1.0, 2.0 ], [ 1.0, 1.0 ]))

    end

end

end
