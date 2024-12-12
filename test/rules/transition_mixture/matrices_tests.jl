@testitem "rules:TransitionMixture:matrices" begin
    using ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions

    import ReactiveMP: @test_rules
    import MatrixCorrectionTools: NoCorrection, AddToDiagonalEntries, ReplaceZeroDiagonalEntries

    @testset "Variational Messages: (q_out_in_switch::Contingency)" begin
        @test_rules [check_type_promotion = false] TransitionMixture{2}((:matrices, k = 1), Marginalisation) [(
            input = (q_out_in_switch = Contingency([0.03 0.06; 0.07 0.14;;; 0.18 0.03; 0.42 0.07]),), output = MatrixDirichlet([1.03 1.18; 1.06 1.03])
        )]

        @test_rules [check_type_promotion = false] TransitionMixture{2}((:matrices, k = 2), Marginalisation) [(
            input = (q_out_in_switch = Contingency([0.03 0.06; 0.07 0.14;;; 0.18 0.03; 0.42 0.07]),), output = MatrixDirichlet([1.07 1.42; 1.1400000000000001 1.07])
        )]
    end
end
