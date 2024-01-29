module ReactiveMPSharedApproximationTest

using ReactiveMP

@testitem "Shared approximation methods" begin
    @testset "__starts_at tests" begin
        import ReactiveMP: __starts_at

        # `@inferred` is need for type-stability
        @test @inferred(__starts_at(((), (), ()))) === (1, 2, 3)
        @test @inferred(__starts_at(((3,), (), ()))) === (1, 4, 5)
        @test @inferred(__starts_at(((), (3,), ()))) === (1, 2, 5)
        @test @inferred(__starts_at(((), (), (3,)))) === (1, 2, 3)
        @test @inferred(__starts_at(((1,), (1,), (1,)))) === (1, 2, 3)
        @test @inferred(__starts_at(((2,), (1,), (1,)))) === (1, 3, 4)
        @test @inferred(__starts_at(((3,), (1,), (1,)))) === (1, 4, 5)
    end

    @testset "__as_vec tests" begin
        import ReactiveMP: __as_vec

        @static if VERSION >= v"1.7"
            # `@inferred` is need for type-stability
            @test @inferred(__as_vec((1, 2, 3))) == [1, 2, 3]
            @test @inferred(__as_vec((3, 2, 1))) == [3, 2, 1]
            @test @inferred(__as_vec((3, [2.0], 1))) == [3.0, 2.0, 1.0]
            @test @inferred(__as_vec(([3.0, 2.0], [2.0], 1))) == [3.0, 2.0, 2.0, 1.0]
            @test @inferred(__as_vec((3, 2, [1.0 0.0; 0.0 1.0]))) == [3.0, 2.0, 1.0, 0.0, 0.0, 1.0]
            @test @inferred(__as_vec(([3.0, 2.0], 2, [1.0 0.0; 0.0 1.0]))) == [3.0, 2.0, 2.0, 1.0, 0.0, 0.0, 1.0]
        else
            # Julia's compiler on <= v1.6 is not smart enough to infer the type of the result
            @test __as_vec((1, 2, 3)) == [1, 2, 3]
            @test __as_vec((3, 2, 1)) == [3, 2, 1]
            @test __as_vec((3, [2.0], 1)) == [3.0, 2.0, 1.0]
            @test __as_vec(([3.0, 2.0], [2.0], 1)) == [3.0, 2.0, 2.0, 1.0]
            @test __as_vec((3, 2, [1.0 0.0; 0.0 1.0])) == [3.0, 2.0, 1.0, 0.0, 0.0, 1.0]
            @test __as_vec(([3.0, 2.0], 2, [1.0 0.0; 0.0 1.0])) == [3.0, 2.0, 2.0, 1.0, 0.0, 0.0, 1.0]
        end
    end

    @testset "__splitjoin tests" begin
        import ReactiveMP: __splitjoin

        # `@inferred` is need for type-stability
        @test @inferred(__splitjoin([1, 2, 3], ((), (), ()))) == (1, 2, 3)
        @test @inferred(__splitjoin([1, 2, 3], ((), (1,), ()))) == (1, [2], 3)
        @test @inferred(__splitjoin([1, 2, 3], ((1,), (1,), ()))) == ([1], [2], 3)
        @test @inferred(__splitjoin([1, 2, 3], ((1,), (1,), (1,)))) == ([1], [2], [3])
        @test @inferred(__splitjoin([1, 2, 1, 0, 0, 1], ((1,), (1,), (2, 2)))) == ([1], [2], [1 0; 0 1])
        @test @inferred(__splitjoin([1, 0, 0, 1, 2, 1, 0, 0, 1], ((2, 2), (1,), (2, 2)))) == ([1 0; 0 1], [2], [1 0; 0 1])
        @test @inferred(__splitjoin([1, 0, 0, 1, 2, 1, 0, 0, 1], ((2, 2), (), (2, 2)))) == ([1 0; 0 1], 2, [1 0; 0 1])
    end
end

end
