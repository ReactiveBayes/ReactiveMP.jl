@testitem "Linearization: sums and differences" tags = [:approximations] begin
    using MessagePassingRulesApproximations

    @test @inferred(approximate(Linearization(), (x, y) -> x + y, (1, 2))) == ([1 1], 0)
    @test @inferred(approximate(Linearization(), (x, y) -> x - y, (1, 2))) == ([1 -1], 0)
    @test @inferred(approximate(Linearization(), (x, y) -> x .- y, ([1.0, 2.0], 1.0))) == ([1.0 0.0 -1.0; 0.0 1.0 -1.0], [0.0, 0.0])
    @test @inferred(approximate(Linearization(), (x, y) -> x .- y, ([1.0, 2.0], [1.0, 1.0]))) == ([1.0 0.0 -1.0 0.0; 0.0 1.0 0.0 -1.0], [0.0, 0.0])
    @test @inferred(approximate(Linearization(), (x) -> x .- [1, 1], (1.0,))) == ([1.0, 1.0], [-1.0, -1.0])
end

@testitem "Linearization: every shape of input and output" tags = [:approximations] begin
    using MessagePassingRulesApproximations, LinearAlgebra

    # A scalar function of a scalar: the tangent line at x̂ = 2, g(x) ≈ g'(x̂) x + b.
    A, b = approximate(Linearization(), x -> x^3, (2.0,))
    @test A ≈ 12.0 && b ≈ 8.0 - 12.0 * 2.0

    # A scalar function of a vector: the gradient as a row.
    A, b = approximate(Linearization(), x -> dot(x, x), ([1.0, 2.0],))
    @test A ≈ [2.0 4.0] && only(A * [1.0, 2.0]) + b ≈ 5.0

    # A vector function of a vector: the Jacobian.
    M = [1.0 2.0; 3.0 4.0]
    A, b = approximate(Linearization(), x -> M * x .+ 1.0, ([0.5, -1.0],))
    @test A ≈ M && b ≈ [1.0, 1.0]

    # Three inputs of mixed shapes: the Jacobian over their concatenation, in input order.
    g(x, y, z) = x .* y .+ z
    A, b = approximate(Linearization(), g, (2.0, [1.0, 3.0], [0.0, 1.0]))
    @test A ≈ [1.0 2.0 0.0 1.0 0.0; 3.0 0.0 2.0 0.0 1.0]
    @test A * [2.0, 1.0, 3.0, 0.0, 1.0] + b ≈ g(2.0, [1.0, 3.0], [0.0, 1.0])

    # The linear map is exact at the expansion point, and a linear function is reproduced.
    A, b = approximate(Linearization(), (x, y) -> 3x - 2y, (0.7, -0.2))
    @test only(A * [0.7, -0.2]) + b ≈ 3 * 0.7 - 2 * -0.2
    @test only(A * [5.0, 1.0]) + b ≈ 13.0

    @test local_linearization(x -> x^2, (3.0,)) == approximate(Linearization(), x -> x^2, (3.0,))
end
