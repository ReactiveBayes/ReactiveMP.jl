
@testitem "rules:MatrixNormal:out" begin
    using ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions

    import ReactiveMP: @test_rules

    @testset "Belief Propagation: (m_M::PointMass, m_U::PointMass, m_V::PointMass)" begin
        @test_rules [check_type_promotion = true] MatrixNormal(
            :out, Marginalisation
        ) [
            (
                input = (
                    m_M = PointMass([1.0 2.0; 3.0 4.0]),
                    m_U = PointMass([2.0 0.0; 0.0 3.0]),
                    m_V = PointMass([1.0 0.0; 0.0 4.0]),
                ),
                output = MatrixNormal(
                    [1.0 2.0; 3.0 4.0], [2.0 0.0; 0.0 3.0], [1.0 0.0; 0.0 4.0]
                ),
            ),
            (
                input = (
                    m_M = PointMass([0.5 1.0; 2.0 3.0; 4.0 5.0]),
                    m_U = PointMass([2.0 0.0 0.0; 0.0 3.0 0.0; 0.0 0.0 4.0]),
                    m_V = PointMass([1.0 0.0; 0.0 2.0]),
                ),
                output = MatrixNormal(
                    [0.5 1.0; 2.0 3.0; 4.0 5.0],
                    [2.0 0.0 0.0; 0.0 3.0 0.0; 0.0 0.0 4.0],
                    [1.0 0.0; 0.0 2.0],
                ),
            ),
            (
                input = (
                    m_M = PointMass(zeros(2, 3)),
                    m_U = PointMass([1.0 0.0; 0.0 1.0]),
                    m_V = PointMass([1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]),
                ),
                output = MatrixNormal(
                    zeros(2, 3),
                    [1.0 0.0; 0.0 1.0],
                    [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0],
                ),
            ),
        ]
    end
end
