
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
end
