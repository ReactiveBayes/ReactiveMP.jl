
@testitem "rules:MatrixNormal:V" begin
    using ReactiveMP,
        BayesBase, Random, ExponentialFamily, Distributions, LinearAlgebra
    using FastCholesky

    import ExponentialFamily: InverseWishartFast
    import ReactiveMP: @test_rules

    @testset "Belief Propagation: (m_out::PointMass, m_M::PointMass, m_U::PointMass)" begin
        @test_rules [check_type_promotion = true] MatrixNormal(
            :V, Marginalisation
        ) [
            (
                input = (
                    m_out = PointMass([1.0 2.0; 3.0 4.0; 5.0 6.0]),
                    m_M = PointMass([0.5 1.0; 2.0 3.0; 4.0 5.0]),
                    m_U = PointMass([2.0 0.0 0.0; 0.0 3.0 0.0; 0.0 0.0 4.0]),
                ),
                output = InverseWishartFast(
                    0.0,
                    let X = [1.0 2.0; 3.0 4.0; 5.0 6.0],
                        M = [0.5 1.0; 2.0 3.0; 4.0 5.0],
                        U = [2.0 0.0 0.0; 0.0 3.0 0.0; 0.0 0.0 4.0]

                        (X - M)' * cholinv(U) * (X - M)
                    end,
                ),
            ),
            (
                input = (
                    m_out = PointMass([2.0 1.0; -1.0 0.5]),
                    m_M = PointMass([1.0 0.0; 0.0 1.0]),
                    m_U = PointMass([2.0 -0.5; -0.5 1.0]),
                ),
                output = InverseWishartFast(
                    -1.0,
                    let X = [2.0 1.0; -1.0 0.5],
                        M = [1.0 0.0; 0.0 1.0],
                        U = [2.0 -0.5; -0.5 1.0]

                        (X - M)' * cholinv(U) * (X - M)
                    end,
                ),
            ),
        ]
    end

    @testset "Mean-field VMP: (q_out::PointMass, q_M::PointMass, q_U::InverseWishart)" begin
        # looser tolerance: the InverseWishart -> low-precision conversion path in the
        # `check_type_promotion` tests accumulates float error through `mean(cholinv, q_U)`.
        @test_rules [
            check_type_promotion = true,
            atol = [Float32 => 1e-3, Float64 => 1e-5, BigFloat => 1e-8],
        ] MatrixNormal(:V, Marginalisation) [(
            input = (
                q_out = PointMass([1.0 2.0; 3.0 4.0; 5.0 6.0]),
                q_M = PointMass([0.5 1.0; 2.0 3.0; 4.0 5.0]),
                q_U = InverseWishart(
                    6.0, [2.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 2.0]
                ),
            ),
            output = InverseWishartFast(
                0.0,
                let X = [1.0 2.0; 3.0 4.0; 5.0 6.0],
                    M = [0.5 1.0; 2.0 3.0; 4.0 5.0],
                    q_U = InverseWishart(
                        6.0, [2.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 2.0]
                    )

                    D = X - M
                    D' * mean(cholinv, q_U) * D
                end,
            ),
        ),]
    end

    @testset "Mean-field VMP: (q_out::MatrixNormal, q_M::PointMass, q_U::InverseWishart)" begin
        @test_rules [
            check_type_promotion = true,
            atol = [Float32 => 1e-3, Float64 => 1e-5, BigFloat => 1e-8],
        ] MatrixNormal(:V, Marginalisation) [(
            input = (
                q_out = MatrixNormal(
                    [1.0 2.0; 3.0 4.0; 5.0 6.0],
                    [0.5 0.0 0.0; 0.0 0.5 0.0; 0.0 0.0 0.5],
                    [1.0 0.0; 0.0 0.5],
                ),
                q_M = PointMass([0.5 1.0; 2.0 3.0; 4.0 5.0]),
                q_U = InverseWishart(
                    6.0, [2.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 2.0]
                ),
            ),
            output = InverseWishartFast(
                0.0,
                let X = [1.0 2.0; 3.0 4.0; 5.0 6.0],
                    M = [0.5 1.0; 2.0 3.0; 4.0 5.0],
                    U_out = [0.5 0.0 0.0; 0.0 0.5 0.0; 0.0 0.0 0.5],
                    V_out = [1.0 0.0; 0.0 0.5],
                    q_U = InverseWishart(
                        6.0, [2.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 2.0]
                    )

                    D = X - M
                    invU = mean(cholinv, q_U)
                    D' * invU * D + tr(invU * U_out) * V_out
                end,
            ),
        ),]
    end
end
