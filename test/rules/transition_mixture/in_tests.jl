@testitem "rules:TransitionMixture:in" begin
    using ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions

    import ReactiveMP: @test_rules
    import MatrixCorrectionTools: NoCorrection, AddToDiagonalEntries, ReplaceZeroDiagonalEntries

    @testset "Belief Propagation: (m_out::Categorical, m_switch::Categorical, m_matrices::ManyOf{N, PointMass})" begin
        @test_rules [check_type_promotion = false] TransitionMixture{2}(:in, Marginalisation) [(
            input = (m_out = Categorical(0.2, 0.8), m_switch = Categorical([1.0, 0.0]), m_matrices = ManyOf(PointMass([0.1 0.2; 0.9 0.8]), PointMass([0.3 0.4; 0.7 0.6]))),
            output = Categorical([0.5211267605633803, 0.4788732394366197])
        )]
    end

    @testset "Variational Messages: (m_out::Categorical, m_switch::Categorical, q_matrices::ManyOf{N, MatrixDirichlet})" begin
        @test_rules [check_type_promotion = false] TransitionMixture{2}(:in, Marginalisation) [
            (
                input = (m_out = Categorical(0.2, 0.8), m_switch = Categorical([0.3, 0.7]), q_matrices = ManyOf(PointMass([0.1 0.2; 0.9 0.8]), PointMass([0.3 0.4; 0.7 0.6]))),
                output = Categorical([0.5239616613418148, 0.4760383386581853])
            ),
            (
                input = (
                    m_out = Categorical(0.1, 0.9), m_switch = Categorical([0.3, 0.7]), q_matrices = ManyOf(MatrixDirichlet([0.1 0.2; 0.9 0.8]), MatrixDirichlet([0.3 0.4; 0.7 0.6]))
                ),
                output = Categorical([0.5641249477901458, 0.4358750522098543])
            )
        ]
    end
end