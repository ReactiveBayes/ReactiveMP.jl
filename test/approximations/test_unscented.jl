module ReactiveMPUnscentedApproximationTest

using Test
using ReactiveMP

import ReactiveMP: Unscented, unscented_statistics

@testset "Unscented approximation method" begin
 
    
    @testset "Univariate `unscented_statistics`" begin 

        # Compared against the ForneyLab implementation
        @test all(unscented_statistics(Unscented(), 0.0, 1.0, (x) -> x) .≈ (0.0, 1.0, 1.0))
        @test all(unscented_statistics(Unscented(), 0.0, 1.0, (x) -> x ^ 2) .≈ (1.0, 2.0, 0.0))
        @test all(unscented_statistics(Unscented(), -3.0, 1.0, (x) -> x ^ 2) .≈ (10., 38., -6.))
        @test all(isapprox.(unscented_statistics(Unscented(), 0.0, 1.0, (x) -> sin(x) + cos(x)), (0.5, 1.5, 1.0); atol = 1e-6))
        @test all(isapprox.(unscented_statistics(Unscented(), -3.0, 3.0, (x) -> sin(x) + cos(x)), (0.56555582, 7.91911820, -2.54661619); atol = 1e-6))

        # Compared against the ForneyLab implementation
        @test all(unscented_statistics(Unscented(alpha = 2, beta = 3, kappa = 4), 0.0, 3.0, (x) -> x) .≈ (0.0, 3.0, 3.0))
        @test all(unscented_statistics(Unscented(alpha = 2, beta = 3, kappa = 4), -1.0, 1.0, (x) -> x ^ 2) .≈ (2.0, 23.0, -2.0))
        @test all(unscented_statistics(Unscented(alpha = 2, beta = 3, kappa = 4), 0.0, 3.0, (x) -> x ^ 2) .≈ (3.0, 171.0, 0.0))
        @test all(isapprox.(unscented_statistics(Unscented(alpha = 2, beta = 3, kappa = 4), 0.0, 1.0, (x) -> sin(x) + cos(x)), (0.9381025804009704, 0.11996354864504076, -0.21718431835123952); atol = 1e-6))
        @test all(isapprox.(unscented_statistics(Unscented(alpha = 2, beta = 3, kappa = 4), -3.0, 3.0, (x) -> sin(x) + cos(x)), (-1.0806538600139532, 0.08398591776750242, -0.32685086597452473); atol = 1e-6))

    end

    @testset "Multivariate `unscented_statistics`" begin 

        # Compared against the ForneyLab implementation
        @test all(unscented_statistics(Unscented(), [ 0.0, 0.0 ], [ 1.0 0.0; 0.0 1.0 ], (x) -> x) .≈ ([0.0, 0.0], [1.0 0.0; 0.0 1.0], [1.0 0.0; 0.0 1.0]))
        @test all(unscented_statistics(Unscented(), [ 2.0, -3.0 ], [ 4.0 -1.0; -1.0 2.0 ], (x) -> x) .≈ ([ 2.0, -3.0 ], [ 4.0 -1.0; -1.0 2.0 ], [ 4.0 -1.0; -1.0 2.0 ]))

        @test all(isapprox.(unscented_statistics(Unscented(), [ 1.0, -1.0 ], [ 1.0 0.0; 0.0 1.0 ], (x) -> x .^ 2), ([2., 2.], [6. 2.; 2. 6.], [2. 0; 0. -2.]), atol = 1e-4))
        @test all(isapprox.(unscented_statistics(Unscented(), [ 2.0, -3.0 ], [ 4.0 -1.0; -1.0 2.0 ], (x) -> x .^ 2), ([8., 11.], [96. 40.; 40. 80.], [16. 6.; -4. -12.]), atol = 1e-4))

        # Compared against the ForneyLab implementation
        @test all(unscented_statistics(Unscented(alpha = 2, beta = 3, kappa = 4), [ 0.0, 0.0 ], [ 1.0 0.0; 0.0 1.0 ], (x) -> x) .≈ ([0.0, 0.0], [1.0 0.0; 0.0 1.0], [1.0 0.0; 0.0 1.0]))
        @test all(unscented_statistics(Unscented(alpha = 2, beta = 3, kappa = 4), [ 2.0, -3.0 ], [ 4.0 -1.0; -1.0 2.0 ], (x) -> x) .≈ ([ 2.0, -3.0 ], [ 4.0 -1.0; -1.0 2.0 ], [ 4.0 -1.0; -1.0 2.0 ]))

        @test all(isapprox.(unscented_statistics(Unscented(alpha = 2, beta = 3, kappa = 4), [ 1.0, -1.0 ], [ 1.0 0.0; 0.0 1.0 ], (x) -> x .^ 2), ([2., 2.], [27. -1.; -1. 27.], [2. 0; 0. -2.]), atol = 1e-4))
        @test all(isapprox.(unscented_statistics(Unscented(alpha = 0.5, beta = 0.1, kappa = 0.2), [ 2.0, -3.0 ], [ 4.0 -1.0; -1.0 2.0 ], (x) -> x .+ 1), ([3., -2.], [ 4.0 -1.0; -1.0 2.0 ], [ 4.0 -1.0; -1.0 2.0 ]), atol = 1e-4))

    end

end

end
