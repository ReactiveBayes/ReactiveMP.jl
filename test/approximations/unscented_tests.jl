
@testitem "Unscented approximation method" begin
    using ReactiveMP

    import ReactiveMP: Unscented, unscented_statistics, approximate

    @testset "Univariate `unscented_statistics`" begin

        # Compared against the ForneyLab implementation
        @test all(unscented_statistics(Unscented(), (x) -> x, (0.0,), (1.0,)) .≈ (0.0, 1.0, 1.0))
        @test all(unscented_statistics(Unscented(), (x) -> x^2, (0.0,), (1.0,)) .≈ (1.0, 2.0, 0.0))
        @test all(unscented_statistics(Unscented(), (x) -> x^2, (-3.0,), (1.0,)) .≈ (10.0, 38.0, -6.0))
        @test all(isapprox.(unscented_statistics(Unscented(), (x) -> sin(x) + cos(x), (0.0,), (1.0,)), (0.5, 1.5, 1.0); atol = 1e-6))
        @test all(isapprox.(unscented_statistics(Unscented(), (x) -> sin(x) + cos(x), (-3.0,), (3.0,)), (0.56555582, 7.91911820, -2.54661619); atol = 1e-6))

        # Compared against the ForneyLab implementation
        @test all(unscented_statistics(Unscented(; alpha = 2, beta = 3, kappa = 4), (x) -> x, (0.0,), (3.0,)) .≈ (0.0, 3.0, 3.0))
        @test all(unscented_statistics(Unscented(; alpha = 2, beta = 3, kappa = 4), (x) -> x^2, (-1.0,), (1.0,)) .≈ (2.0, 23.0, -2.0))
        @test all(unscented_statistics(Unscented(; alpha = 2, beta = 3, kappa = 4), (x) -> x^2, (0.0,), (3.0,)) .≈ (3.0, 171.0, 0.0))
        @test all(
            isapprox.(
                unscented_statistics(Unscented(; alpha = 2, beta = 3, kappa = 4), (x) -> sin(x) + cos(x), (0.0,), (1.0,)),
                (0.9381025804009704, 0.11996354864504076, -0.21718431835123952);
                atol = 1e-6
            )
        )
        @test all(
            isapprox.(
                unscented_statistics(Unscented(; alpha = 2, beta = 3, kappa = 4), (x) -> sin(x) + cos(x), (-3.0,), (3.0,)),
                (-1.0806538600139532, 0.08398591776750242, -0.32685086597452473);
                atol = 1e-6
            )
        )
    end

    @testset "Multivariate `unscented_statistics`" begin

        # Compared against the ForneyLab implementation
        @test all(unscented_statistics(Unscented(), (x) -> x, ([0.0, 0.0],), ([1.0 0.0; 0.0 1.0],)) .≈ ([0.0, 0.0], [1.0 0.0; 0.0 1.0], [1.0 0.0; 0.0 1.0]))
        @test all(unscented_statistics(Unscented(), (x) -> x, ([2.0, -3.0],), ([4.0 -1.0; -1.0 2.0],)) .≈ ([2.0, -3.0], [4.0 -1.0; -1.0 2.0], [4.0 -1.0; -1.0 2.0]))

        @test all(
            isapprox.(unscented_statistics(Unscented(), (x) -> x .^ 2, ([1.0, -1.0],), ([1.0 0.0; 0.0 1.0],)), ([2.0, 2.0], [6.0 2.0; 2.0 6.0], [2.0 0; 0.0 -2.0]), atol = 1e-4)
        )
        @test all(
            isapprox.(
                unscented_statistics(Unscented(), (x) -> x .^ 2, ([2.0, -3.0],), ([4.0 -1.0; -1.0 2.0],)),
                ([8.0, 11.0], [96.0 40.0; 40.0 80.0], [16.0 6.0; -4.0 -12.0]),
                atol = 1e-4
            )
        )

        # Compared against the ForneyLab implementation
        @test all(
            unscented_statistics(Unscented(; alpha = 2, beta = 3, kappa = 4), (x) -> x, ([0.0, 0.0],), ([1.0 0.0; 0.0 1.0],)) .≈
            ([0.0, 0.0], [1.0 0.0; 0.0 1.0], [1.0 0.0; 0.0 1.0])
        )
        @test all(
            unscented_statistics(Unscented(; alpha = 2, beta = 3, kappa = 4), (x) -> x, ([2.0, -3.0],), ([4.0 -1.0; -1.0 2.0],)) .≈
            ([2.0, -3.0], [4.0 -1.0; -1.0 2.0], [4.0 -1.0; -1.0 2.0])
        )

        @test all(
            isapprox.(
                unscented_statistics(Unscented(; alpha = 2, beta = 3, kappa = 4), (x) -> x .^ 2, ([1.0, -1.0],), ([1.0 0.0; 0.0 1.0],)),
                ([2.0, 2.0], [27.0 -1.0; -1.0 27.0], [2.0 0; 0.0 -2.0]),
                atol = 1e-4
            )
        )
        @test all(
            isapprox.(
                unscented_statistics(Unscented(; alpha = 0.5, beta = 0.1, kappa = 0.2), (x) -> x .+ 1, ([2.0, -3.0],), ([4.0 -1.0; -1.0 2.0],)),
                ([3.0, -2.0], [4.0 -1.0; -1.0 2.0], [4.0 -1.0; -1.0 2.0]),
                atol = 1e-4
            )
        )
    end

    @testset "Univariate approximate `unscented_statistics` dispatch `statistic_estimation`" begin
        @test all(approximate(Unscented(), (x) -> x .- [1, 1], (1.0,), (1.0,)) .≈ ([0.0, 0.0], [1.0 1.0; 1.0 1.0]))
        @test all(approximate(Unscented(), (x) -> [x^2, x], (1.0,), (1.0,)) .≈ ([2.0, 1.0], [6.0 2.0; 2.0 1.0]))
    end

    @testset "Unscented approximate input edge cases" begin
        @test_throws DomainError approximate(Unscented(), (x) -> x + 1.0 , (1.0,), (Inf,))
        @test all(approximate(Unscented(), (x) -> x + 1.0, (1.0,), (0.0,) .≈ (2.0, 0.0)))
        @test all(approximate(Unscented(), (x) -> [x^2, x], (2.0,), (0.0,)) .≈ ([4.0, 2.0], [0.0 0.0; 0.0 0.0]))
        @test all(approximate(Unscented(), (x) -> x + 1.0, (NaN,), (1.0,) .≈ (NaN, NaN)))
        @test all(approximate(Unscented(), (x) -> [x^2, x], (NaN,), (1.0,)) .≈ ([NaN, NaN], [NaN NaN; NaN NaN]))
        @test all(approximate(Unscented(), (x) -> x + 1.0, (1.0,), (NaN,) .≈ (NaN, NaN)))
    end

end


