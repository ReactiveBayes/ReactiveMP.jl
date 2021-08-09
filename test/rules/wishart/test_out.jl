module RulesWishartOutTest

using Test
using ReactiveMP
using Random
using Distributions

import ReactiveMP: @test_rules


@testset "rules:Wishart:out" begin

    @testset "Belief Propagation: (m_ν::PointMass, m_S::PointMass)" begin
        
        @test_rules [ with_float_conversions = true ] Wishart(:out, Marginalisation) [
            (input = (m_ν = PointMass(2.0), m_S = PointMass([ 1.0 0.0; 0.0 1.0 ])), output = Wishart(2.0, [ 1.0 0.0; 0.0 1.0 ])),
            (input = (m_ν = PointMass(3.0), m_S = PointMass([ 10.0 -1.0; -1.0 3.0 ])), output = Wishart(3.0, [ 10.0 -1.0; -1.0 3.0 ])),
            (input = (m_ν = PointMass(4.0), m_S = PointMass([ 1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0 ])), output = Wishart(4.0, [ 1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0 ])),
        ]

    end

    @testset "Variational: (m_ν::PointMass, q_S::Any)" begin

        @test_rules [ with_float_conversions = true ] Wishart(:out, Marginalisation) [
            (input = (m_ν = PointMass(2.0), q_S = PointMass([ 1.0 0.0; 0.0 1.0 ])), output = Wishart(2.0, [ 1.0 0.0; 0.0 1.0 ])),
            (input = (m_ν = PointMass(3.0), q_S = PointMass([ 10.0 -1.0; -1.0 3.0 ])), output = Wishart(3.0, [ 10.0 -1.0; -1.0 3.0 ])),
            (input = (m_ν = PointMass(4.0), q_S = PointMass([ 1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0 ])), output = Wishart(4.0, [ 1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0 ])),
        ]
        
    end

    @testset "Variational: (q_ν::Any, m_S::PointMass)" begin

        @test_rules [ with_float_conversions = true ] Wishart(:out, Marginalisation) [
            (input = (q_ν = PointMass(2.0), m_S = PointMass([ 1.0 0.0; 0.0 1.0 ])), output = Wishart(2.0, [ 1.0 0.0; 0.0 1.0 ])),
            (input = (q_ν = PointMass(3.0), m_S = PointMass([ 10.0 -1.0; -1.0 3.0 ])), output = Wishart(3.0, [ 10.0 -1.0; -1.0 3.0 ])),
            (input = (q_ν = PointMass(4.0), m_S = PointMass([ 1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0 ])), output = Wishart(4.0, [ 1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0 ])),
        ]
        
    end

    @testset "Variational: (m_ν::Any, m_S::Any)" begin

        @test_rules [ with_float_conversions = true ] Wishart(:out, Marginalisation) [
            (input = (q_ν = PointMass(2.0), q_S = PointMass([ 1.0 0.0; 0.0 1.0 ])), output = Wishart(2.0, [ 1.0 0.0; 0.0 1.0 ])),
            (input = (q_ν = PointMass(3.0), q_S = PointMass([ 10.0 -1.0; -1.0 3.0 ])), output = Wishart(3.0, [ 10.0 -1.0; -1.0 3.0 ])),
            (input = (q_ν = PointMass(4.0), q_S = PointMass([ 1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0 ])), output = Wishart(4.0, [ 1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0 ])),
        ]
        
    end

end



end