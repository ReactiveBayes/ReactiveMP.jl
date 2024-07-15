@testitem "rules:TransitionMixture:switch" begin
    using ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions

    import ReactiveMP: @test_rules
    import MatrixCorrectionTools: NoCorrection, AddToDiagonalEntries, ReplaceZeroDiagonalEntries

    @testset "Belief Propagation: (m_out::Categorical, m_in::Categorical, m_matrices::ManyOf{N, PointMass})" begin
        @test_rules [check_type_promotion = false] TransitionMixture{2}(:switch, Marginalisation) [(
            input = (m_out = Categorical(0.2, 0.8), m_in = Categorical([1.0, 0.0]), m_matrices = ManyOf(PointMass([0.1 0.2; 0.9 0.8]), PointMass([0.3 0.4; 0.7 0.6]))),
            output = Categorical([0.5441176470588237, 0.4558823529411764])
        )]
    end

    @testset "Variational Messages: (m_out::Categorical, m_in::Categorical, q_matrices::ManyOf{N, MatrixDirichlet})" begin
        @test_rules [check_type_promotion = false] TransitionMixture{2}(:switch, Marginalisation) [
            (
                input = (m_out = Categorical(0.2, 0.8), m_in = Categorical([0.1, 0.9]), q_matrices = ManyOf(PointMass([0.1 0.2; 0.9 0.8]), PointMass([0.3 0.4; 0.7 0.6]))),
                output = Categorical([0.5479233226837061, 0.4520766773162939])
            ),
            (
                input = (
                    m_out = Categorical(0.1, 0.9), m_in = Categorical([0.3, 0.7]), q_matrices = ManyOf(MatrixDirichlet([0.1 0.2; 0.9 0.8]), MatrixDirichlet([0.3 0.4; 0.7 0.6]))
                ),
                output = Categorical(0.6243369962724196, 0.37566300372758055)
            )
        ]
    end
end