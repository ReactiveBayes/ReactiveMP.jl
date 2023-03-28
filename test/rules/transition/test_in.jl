module RulesTransitionInTest

using Test
using ReactiveMP
using Random

import ReactiveMP: @test_rules

@testset "rules:Transition:in" begin
    @testset "Belief Propagation: (m_out::Categorical, m_a::PointMass)" begin
        @test_rules [with_float_conversions = false] Transition(:in, Marginalisation) [(
            input = (m_out = Categorical([0.1, 0.4, 0.5]), m_a = PointMass([0.2 0.1 0.7; 0.4 0.3 0.3; 0.1 0.6 0.3])),
            output = Categorical([0.23000000000000004, 0.43, 0.33999999999999997])
        )]
    end

    @testset "Variational Bayes: (q_out::Any, q_a::MatrixDirichlet)" begin
        @test_rules [with_float_conversions = false] Transition(:in, Marginalisation) [(
            input = (q_out = PointMass([0.1, 0.4, 0.5]), q_a = MatrixDirichlet([0.2 0.1 0.7; 0.4 0.3 0.3; 0.1 0.6 0.3])),
            output = Categorical([0.03245589526827472, 0.5950912160314408, 0.37245288870028453])
        )]
    end

    @testset "Variational Bayes: (m_out::Categorical, q_a::MatrixDirichlet)" begin
        @test_rules [with_float_conversions = false] Transition(:in, Marginalisation) [
            (
                input = (m_out = Categorical([0.1, 0.4, 0.5]), q_a = MatrixDirichlet([0.2 0.1 0.7; 0.4 0.3 0.3; 0.1 0.6 0.3])),
                output = Categorical([0.27575594149188243, 0.5503434892576381, 0.17390056925047945])
            ),
            (
                input = (m_out = Categorical([0.3, 0.3, 0.4]), q_a = MatrixDirichlet(diageye(3) .+ 1)),
                output = Categorical([0.32119415576170857, 0.32119415576170857, 0.357611688476583])
            )
        ]
    end

    @testset "Variational Bayes: (m_out::Categorical, q_a::PointMass)" begin
        @test_rules [with_float_conversions = false] Transition(:in, Marginalisation) [(
            input = (m_out = Categorical([0.1, 0.4, 0.5]), q_a = PointMass([0.2 0.1 0.7; 0.4 0.3 0.3; 0.1 0.6 0.3])),
            output = Categorical([0.23000000000000004, 0.43, 0.33999999999999997])
        )]
    end

    @testset "Variational Bayes: (q_out::PointMass, q_a::PointMass)" begin
        @test_rules [with_float_conversions = false] Transition(:in, Marginalisation) [(
            input = (q_out = PointMass([0.1, 0.4, 0.5]), q_a = PointMass([0.2 0.1 0.7; 0.4 0.3 0.3; 0.1 0.6 0.3])),
            output = Categorical([0.23000000000000004, 0.43, 0.33999999999999997])
        )]
    end
end

end
