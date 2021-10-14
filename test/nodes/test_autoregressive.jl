module AutoregressiveNodeTest

using Test
using ReactiveMP
using Random

import ReactiveMP: @test_rules

@testset "AutoregressiveNode" begin

    @testset "ARTransitionMatrix" begin
        
        rng = MersenneTwister(1233)

        for γ in randn(rng, 3), order in 2:4
            transition = ReactiveMP.ARTransitionMatrix(order, γ)
            matrix      = rand(rng, order, order)
            ftransition = zeros(order, order)
            ftransition[1] = inv(γ)

            @test broadcast(+, matrix, transition) == (matrix + ftransition)
            @test_throws DimensionMismatch broadcast(+, zeros(order + 1, order + 1), transition)

            @test ReactiveMP.add_transition(matrix, transition) == (matrix + ftransition)
            @test_throws DimensionMismatch ReactiveMP.add_transition(zeros(order + 1, order + 1), transition)
            
            cmatrix = copy(matrix)
            broadcast!(+, cmatrix, transition)
            @test cmatrix == (matrix + ftransition)

            cmatrix = copy(matrix)
            ReactiveMP.add_transition!(cmatrix, transition)
            @test cmatrix == (matrix + ftransition)
        end

    end
end

end
