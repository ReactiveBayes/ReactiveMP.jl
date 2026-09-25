@testitem "algebra:PermutationMatrix" tags = [:algebra] begin
    using FlowMessagePassingRules, LinearAlgebra, Random, StableRNGs
    using FlowMessagePassingRules: getind, PT_X_P

    # Ones at (k, ind[k]); multiplying permutes.
    P = PermutationMatrix([2, 1, 3])
    @test P == [0 1 0; 1 0 0; 0 0 1]
    @test size(P) == (3, 3) && size(P, 1) == 3 && size(P, 3) == 1 && length(P) == 9
    @test eltype(P) == Int
    @test sum(P) == 3 && all(k -> sum(P[:, k]) == 1 && sum(P[k, :]) == 1, 1:3)
    ind = shuffle(StableRNG(3), collect(1:100))
    @test getind(PermutationMatrix(ind)) == ind
    @test getind(PermutationMatrix(ind)') == sortperm(ind)
    @test getind(transpose(PermutationMatrix(ind))) == sortperm(ind)

    # A random one, from the generator given or the task's; the first index always moves,
    # unless asked otherwise.
    for dim in (2, 3, 5, 10)
        Q = PermutationMatrix(StableRNG(dim), dim)
        @test Q == PermutationMatrix(StableRNG(dim), dim)
        @test Q[1, 1] == 0
        @test sort(Q.ind) == 1:dim
        @test PermutationMatrix(dim) isa PermutationMatrix{Int}
    end
    @test any(seed -> PermutationMatrix(StableRNG(seed), 3; switch_first = false)[1, 1] == 1, 1:100)
    @test inv(P) == P'

    # Against the dense matrix: vectors, row vectors and square matrices, with the adjoint and
    # the transpose on either side.
    rng = StableRNG(1)
    for dim in (2, 3, 5, 50)
        Q = PermutationMatrix(StableRNG(10 + dim), dim)
        D = Matrix(Q)
        x, X = randn(rng, dim), randn(rng, dim, dim)
        for (A, dense) in ((Q, D), (Q', D'), (transpose(Q), transpose(D)))
            @test A * x == dense * x
            for Z in (X, X', transpose(X))
                @test A * Z == dense * Z
                @test Z * A == Z * dense
            end
            @test x' * A == x' * dense
            @test transpose(x) * A == transpose(x) * dense
            y, Y = similar(x), similar(X)
            mul!(y, A, x)
            @test y == dense * x
            mul!(Y, A, X)
            @test Y == dense * X
            mul!(Y, X, A)
            @test Y == X * dense
        end
        @test PT_X_P(X, Q) == D' * X * D
    end
end

@testitem "algebra:PermutationMatrix, a product of two" tags = [:algebra] begin
    using FlowMessagePassingRules, LinearAlgebra, StableRNGs

    # A product of two permutations is a permutation, with the indices composed.
    P, Q = PermutationMatrix(StableRNG(1), 4), PermutationMatrix(StableRNG(2), 4)
    for A in (P, P', transpose(P)), B in (Q, Q', transpose(Q))
        product = A * B
        @test product isa PermutationMatrix
        @test product == Matrix(A) * Matrix(B)
        Y = zeros(Int, 4, 4)
        mul!(Y, A, B)
        @test Y == Matrix(A) * Matrix(B)
    end
end
