@testitem "rules:TransitionMixture:out" begin
    using ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions

    import ReactiveMP: @test_rules
    import MatrixCorrectionTools: NoCorrection, AddToDiagonalEntries, ReplaceZeroDiagonalEntries

    @testset "Belief Propagation: (m_in::Categorical, m_switch::Categorical, m_matrices::ManyOf{N, PointMass})" begin
        @test_rules [check_type_promotion = false] TransitionMixture{2}(:out, Marginalisation) [(
            input = (m_in = Categorical(0.1, 0.9), m_switch = Categorical([1.0, 0.0]), m_matrices = ManyOf(PointMass([0.1 0.2; 0.9 0.8]), PointMass([0.3 0.4; 0.7 0.6]))),
            output = Categorical([0.19, 0.81])
        )]
    end

    @testset "Variational Messages: (m_in::Categorical, m_switch::Categorical, q_matrices::ManyOf{N, MatrixDirichlet})" begin
        @test_rules [check_type_promotion = false] TransitionMixture{2}(:out, Marginalisation) [
            (
                input = (m_in = Categorical(0.1, 0.9), m_switch = Categorical([0.3, 0.7]), q_matrices = ManyOf(PointMass([0.1 0.2; 0.9 0.8]), PointMass([0.3 0.4; 0.7 0.6]))),
                output = Categorical([0.33000000000033997, 0.6699999999996599])
            ),
            (
                input = (
                    m_in = Categorical(0.1, 0.9), m_switch = Categorical([0.3, 0.7]), q_matrices = ManyOf(MatrixDirichlet([0.1 0.2; 0.9 0.8]), MatrixDirichlet([0.3 0.4; 0.7 0.6]))
                ),
                output = Categorical([0.16046880450644171, 0.8395311954935583])
            )
        ]
    end
end