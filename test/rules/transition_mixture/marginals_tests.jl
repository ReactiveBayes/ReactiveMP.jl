@testitem "marginalrules:TransitionMixture" begin
    using ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions

    import ReactiveMP: @test_marginalrules
    @testset "out_in_switch: (m_out::Categorical, m_in::Categorical, m_switch::Categorical, q_matrices::ManyOf{N, Union{PointMass, MatrixDirichlet}})" begin
        @test_marginalrules [check_type_promotion = false] TransitionMixture{2}(:out_in_switch) [(
            input = (
                m_out = Categorical(0.2, 0.8),
                m_in = Categorical(0.1, 0.9),
                m_switch = Categorical(0.3, 0.7),
                q_matrices = ManyOf(PointMass([0.1 0.2; 0.9 0.8]), PointMass([0.3 0.4; 0.7 0.6]))
            ),
            output = Contingency(
                permutedims(
                    stack([
                        [0.0009966777408637875 0.017940199335548173; 0.035880398671096346 0.28704318936877077],
                        [0.0069767441860465115 0.08372093023255814; 0.06511627906976744 0.5023255813953489]
                    ]),
                    (3, 1, 2)
                )
            )
        )]
    end
end