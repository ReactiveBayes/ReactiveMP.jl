
@testitem "rules:MatrixNormalWishart:out" begin
    using ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions

    import ReactiveMP: @test_rules

    @testset "Variational Message Passing: (q_M::PointMass, q_U::PointMass, q_V::PointMass, q_ν::PointMass)" begin
        # Type promotion is disabled (mirrors MvNormalWishart): ν is a scalar and would not
        # necessarily promote the matrix parameters M, U, V.
        @test_rules [check_type_promotion = false] MatrixNormalWishart(
            :out, Marginalisation
        ) [
            (
                input = (
                    q_M = PointMass([1.0 2.0; 3.0 4.0]),
                    q_U = PointMass([2.0 0.0; 0.0 2.0]),
                    q_V = PointMass([1.0 0.0; 0.0 1.0]),
                    q_ν = PointMass(5.0),
                ),
                output = MatrixNormalWishart(
                    [1.0 2.0; 3.0 4.0], [2.0 0.0; 0.0 2.0], [1.0 0.0; 0.0 1.0], 5.0
                ),
            ),
            (
                input = (
                    q_M = PointMass([0.5 1.0; 2.0 3.0; 4.0 5.0]),
                    q_U = PointMass([2.0 0.0 0.0; 0.0 3.0 0.0; 0.0 0.0 4.0]),
                    q_V = PointMass([1.0 0.0; 0.0 2.0]),
                    q_ν = PointMass(4.0),
                ),
                output = MatrixNormalWishart(
                    [0.5 1.0; 2.0 3.0; 4.0 5.0],
                    [2.0 0.0 0.0; 0.0 3.0 0.0; 0.0 0.0 4.0],
                    [1.0 0.0; 0.0 2.0],
                    4.0,
                ),
            ),
        ]
    end
end
