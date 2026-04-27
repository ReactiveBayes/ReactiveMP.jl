
@testitem "rules:MatrixNormal:U" begin
    using ReactiveMP,
        BayesBase, Random, ExponentialFamily, Distributions, LinearAlgebra
    using FastCholesky

    import ExponentialFamily: InverseWishartFast
    import ReactiveMP: @test_rules

    @testset "Belief Propagation: (m_out::PointMass, m_M::PointMass, m_V::PointMass)" begin
        @test_rules [check_type_promotion = true] MatrixNormal(
            :U, Marginalisation
        ) [
            (
                input = (
                    m_out = PointMass([1.0 2.0; 3.0 4.0; 5.0 6.0]),
                    m_M = PointMass([0.5 1.0; 2.0 3.0; 4.0 5.0]),
                    m_V = PointMass([1.0 0.0; 0.0 2.0]),
                ),
                output = InverseWishartFast(
                    -2.0,
                    let X = [1.0 2.0; 3.0 4.0; 5.0 6.0],
                        M = [0.5 1.0; 2.0 3.0; 4.0 5.0],
                        V = [1.0 0.0; 0.0 2.0]

                        (X - M) * cholinv(V) * (X - M)'
                    end,
                ),
            ),
            (
                input = (
                    m_out = PointMass([2.0 1.0; -1.0 0.5]),
                    m_M = PointMass([1.0 0.0; 0.0 1.0]),
                    m_V = PointMass([2.0 -0.5; -0.5 1.0]),
                ),
                output = InverseWishartFast(
                    -1.0,
                    let X = [2.0 1.0; -1.0 0.5],
                        M = [1.0 0.0; 0.0 1.0],
                        V = [2.0 -0.5; -0.5 1.0]

                        (X - M) * cholinv(V) * (X - M)'
                    end,
                ),
            ),
        ]
    end
end
