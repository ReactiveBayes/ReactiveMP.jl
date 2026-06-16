
@testitem "marginalrules:MatrixNormal" begin
    using ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions

    import ReactiveMP: @test_marginalrules

    @testset ":out_M_U_V (m_out::PointMass, m_M::PointMass, m_U::PointMass, m_V::PointMass)" begin
        @test_marginalrules [check_type_promotion = true] MatrixNormal(
            :out_M_U_V
        ) [
            (
                input = (
                    m_out = PointMass([1.0 2.0; 3.0 4.0]),
                    m_M = PointMass([0.5 1.0; 1.5 2.0]),
                    m_U = PointMass([2.0 0.0; 0.0 3.0]),
                    m_V = PointMass([1.0 0.0; 0.0 4.0]),
                ),
                output = (
                    out = PointMass([1.0 2.0; 3.0 4.0]),
                    M = PointMass([0.5 1.0; 1.5 2.0]),
                    U = PointMass([2.0 0.0; 0.0 3.0]),
                    V = PointMass([1.0 0.0; 0.0 4.0]),
                ),
            ),
            (
                input = (
                    m_out = PointMass([0.5 1.0; 2.0 3.0; 4.0 5.0]),
                    m_M = PointMass(zeros(3, 2)),
                    m_U = PointMass([2.0 0.0 0.0; 0.0 3.0 0.0; 0.0 0.0 4.0]),
                    m_V = PointMass([1.0 0.0; 0.0 2.0]),
                ),
                output = (
                    out = PointMass([0.5 1.0; 2.0 3.0; 4.0 5.0]),
                    M = PointMass(zeros(3, 2)),
                    U = PointMass([2.0 0.0 0.0; 0.0 3.0 0.0; 0.0 0.0 4.0]),
                    V = PointMass([1.0 0.0; 0.0 2.0]),
                ),
            ),
        ]
    end
end
