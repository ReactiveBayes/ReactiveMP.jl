module ReactiveMPPermutationMatrixTest

using Test
using ReactiveMP
using ReactiveMP: getind, PT_X_P
using Random

using LinearAlgebra

@testset "Permutation Matrix" begin
    @testset "Creation" begin
        ind = shuffle(collect(1:100))
        P = PermutationMatrix(ind)
        @test P.ind == ind

        for k in 2:10
            P = PermutationMatrix(k)
            @test length(P.ind) == k
        end

        # check whether the first entry is one
        for k in 1:100
            P = PermutationMatrix(3)
            @test P[1, 1] == 0
        end
        ind = 0
        for k in 1:1000
            P = PermutationMatrix(3; switch_first = false)
            if P[1, 1] == 1
                ind += 1
            end
        end
        @test ind > 0
    end

    @testset "Get functions" begin
        ind = shuffle(collect(1:100))
        P = PermutationMatrix(ind)
        @test P.ind == getind(P)

        ind = shuffle(collect(1:100))
        P = PermutationMatrix(ind)'
        @test sortperm(ind) == getind(P)
    end

    @testset "Simple base functions" begin
        P = PermutationMatrix(10)
        @test eltype(P) == Int64
        @test length(P) == 100
        @test size(P) == (10, 10)

        @test sum(P) == 10
        for k in 1:10
            @test sum(P[:, k]) == 1
            @test sum(P[k, :]) == 1
        end

        @test inv(P) == P'
    end

    @testset "Multiplication" begin

        # generate matrices
        P = PermutationMatrix(50)
        P_dense = collect(P)
        X = randn(50, 50)
        x = randn(50)

        @test P * x == P_dense * x
        @test P' * x == P_dense' * x
        @test transpose(P) * x == transpose(P_dense) * x

        @test P * X == P_dense * X
        @test P' * X == P_dense' * X
        @test transpose(P) * X == transpose(P_dense) * X

        @test P * X' == P_dense * X'
        @test P' * X' == P_dense' * X'
        @test transpose(P) * X' == transpose(P_dense) * X'

        @test P * transpose(X) == P_dense * transpose(X)
        @test P' * transpose(X) == P_dense' * transpose(X)
        @test transpose(P) * transpose(X) == transpose(P_dense) * transpose(X)

        @test x' * P == x' * P_dense
        @test x' * P' == x' * P_dense'
        @test x' * transpose(P) == x' * transpose(P_dense)

        @test transpose(x) * P == transpose(x) * P_dense
        @test transpose(x) * P' == transpose(x) * P_dense'
        @test transpose(x) * transpose(P) == transpose(x) * transpose(P_dense)

        @test P * X == P_dense * X
        @test P' * X == P_dense' * X
        @test transpose(P) * X == transpose(P_dense) * X

        @test P * X' == P_dense * X'
        @test P' * X' == P_dense' * X'
        @test transpose(P) * X' == transpose(P_dense) * X'

        @test P * transpose(X) == P_dense * transpose(X)
        @test P' * transpose(X) == P_dense' * transpose(X)
        @test transpose(P) * transpose(X) == transpose(P_dense) * transpose(X)

        @test X * P == X * P_dense
        @test X * P' == X * P_dense'
        @test X * transpose(P) == X * transpose(P_dense)

        @test X' * P == X' * P_dense
        @test X' * P' == X' * P_dense'
        @test X' * transpose(P) == X' * transpose(P_dense)

        @test transpose(X) * P == transpose(X) * P_dense
        @test transpose(X) * P' == transpose(X) * P_dense'
        @test transpose(X) * transpose(P) == transpose(X) * transpose(P_dense)

        @test PT_X_P(X, P) == P' * X * P
        @test PT_X_P(X, P) == P_dense' * X * P_dense
    end
end
end
