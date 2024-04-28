
@testitem "marginalrules:InverseWishart" begin
    using ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions

    import ExponentialFamily: InverseWishartFast
    import ReactiveMP: @test_marginalrules

    @testset ":out_ν_S (m_out::InverseWishart, m_ν::PointMass, m_S::PointMass)" begin
        @test_marginalrules [check_type_promotion = true] InverseWishart(:out_ν_S) [
            (
                input = (m_out = InverseWishartFast(3.0, [3.0 -1.0; -1.0 4.0]), m_ν = PointMass(2.0), m_S = PointMass([1.0 0.0; 0.0 1.0])),
                output = (out = InverseWishart(8.0, [4.0 -1.0; -1.0 5.0]), ν = PointMass(2.0), S = PointMass([1.0 0.0; 0.0 1.0]))
            ),
            (
                input = (
                    m_out = InverseWishartFast(4.0, [9.0 -2.0 1.0; -2.0 5.0 -2.0; 1.0 -2.0 11.0]),
                    m_ν = PointMass(3.0),
                    m_S = PointMass([11.0 -2.0 1.0; -2.0 5.0 -2.0; 1.0 -2.0 9.0])
                ),
                output = (
                    out = InverseWishart(11.0, [20.0 -4.0 2.0; -4.0 10.0 -4.0; 2.0 -4.0 20.0]), ν = PointMass(3.0), S = PointMass([11.0 -2.0 1.0; -2.0 5.0 -2.0; 1.0 -2.0 9.0])
                )
            )
        ]
    end
end
