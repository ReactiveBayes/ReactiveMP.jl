module RulesTransitionOutTest

using ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions

import ReactiveMP: @test_rules

@testitem "rules:Transition:out" begin
    @testset "Belief Propagation: (q_in::PointMass, q_a::PointMass)" begin
        @test_rules [check_type_promotion = false] Transition(:out, Marginalisation) [
            (
                input = (q_in = PointMass([0.1, 0.4, 0.5]), q_a = PointMass([0.2 0.1 0.7; 0.4 0.3 0.3; 0.1 0.6 0.3])),
                output = Categorical([0.3660714285714285, 0.27678571428571425, 0.35714285714285715])
            ),
            (
                input = (q_in = PointMass([0.2, 0.5, 0.3]), q_a = PointMass([0.1 0.8 0.1; 0.6 0.3 0.1; 0.2 0.4 0.4])),
                output = Categorical([0.40540540540540543, 0.2702702702702703, 0.32432432432432434])
            )
        ]
    end

    @testset "Belief Propagation: (q_in::Categorical, q_a::PointMass)" begin
        @test_rules [check_type_promotion = false] Transition(:out, Marginalisation) [
            (
                input = (m_in = Categorical([0.1, 0.4, 0.5]), m_a = PointMass([0.2 0.1 0.7; 0.4 0.3 0.3; 0.1 0.6 0.3])),
                output = Categorical([0.3660714285714285, 0.27678571428571425, 0.35714285714285715])
            ),
            (
                input = (m_in = Categorical([0.2, 0.5, 0.3]), m_a = PointMass([0.1 0.8 0.1; 0.6 0.3 0.1; 0.2 0.4 0.4])),
                output = Categorical([0.40540540540540543, 0.2702702702702703, 0.32432432432432434])
            )
        ]
    end

    @testset "Variational Bayes: (q_in::Categorical, q_a::Any)" begin
        @test_rules [check_type_promotion = false] Transition(:out, Marginalisation) [(
            input = (q_in = Categorical([0.1, 0.4, 0.5]), q_a = PointMass([0.2 0.1 0.7; 0.4 0.3 0.3; 0.1 0.6 0.3])),
            output = Categorical([0.2994385327821292, 0.3260399327754116, 0.37452153444245917])
        )]
    end

    @testset "Variational Bayes: (m_in::Categorical, q_a::MatrixDirichlet)" begin
        @test_rules [check_type_promotion = false] Transition(:out, Marginalisation) [
            (input = (q_in = Categorical([0.1, 0.9]), q_a = MatrixDirichlet([0.1 0.1; 0.3 0.4])), output = Categorical([0.0004227712570263357, 0.9995772287429735])),
            (
                input = (q_in = Categorical([0.2, 0.5, 0.3]), q_a = MatrixDirichlet([0.1 0.8 0.1; 0.6 0.3 0.1; 0.2 0.4 0.4])),
                output = Categorical([0.0626653372827384, 0.10413392154490132, 0.8332007411723602])
            )
        ]
    end

    @testset "Variational Bayes: (m_in::DiscreteNonParametric, q_a::PointMass)" begin
        @test_rules [check_type_promotion = false] Transition(:out, Marginalisation) [
            (
                input = (m_in = Categorical([0.1, 0.4, 0.5]), q_a = PointMass([0.2 0.1 0.7; 0.4 0.3 0.3; 0.1 0.6 0.3])),
                output = Categorical([0.3660714285714285, 0.27678571428571425, 0.35714285714285715])
            ),
            (
                input = (m_in = Categorical([0.2, 0.5, 0.3]), q_a = PointMass([0.1 0.8 0.1; 0.6 0.3 0.1; 0.2 0.4 0.4])),
                output = Categorical([0.40540540540540543, 0.2702702702702703, 0.32432432432432434])
            )
        ]
    end
end

end
