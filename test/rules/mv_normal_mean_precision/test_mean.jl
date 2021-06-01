module RulesMvNormalMeanPrecisionMeanTest

using Test
using ReactiveMP
using Random

import ReactiveMP: @test_rules

@testset "rules:MvNormalMeanPrecision:mean" begin

    @testset "Belief Propagation: (m_out::PointMass, m_Λ::PointMass)" begin
        
        @test_rules [ with_float_conversions = true ] MvNormalMeanPrecision(:μ, Marginalisation) [
            (input = (m_out = PointMass([ -1.0 ]), m_Λ = PointMass([ 2.0 ])), output = MvNormalMeanPrecision([ -1.0 ], [ 2.0 ])),
            (input = (m_out = PointMass([  1.0 ]), m_Λ = PointMass([ 2.0 ])), output = MvNormalMeanPrecision([  1.0 ], [ 2.0 ])),
            (input = (m_out = PointMass([ 2.0 ]),  m_Λ = PointMass([ 1.0 ])), output = MvNormalMeanPrecision([  2.0 ], [ 1.0 ])),
            (input = (m_out = PointMass([ 1.0, 3.0 ]),  m_Λ = PointMass([ 3.0 2.0; 2.0 4.0 ])),   output = MvNormalMeanPrecision([ 1.0, 3.0 ], [ 3.0 2.0; 2.0 4.0 ])),
            (input = (m_out = PointMass([ -1.0, 2.0 ]), m_Λ = PointMass([ 7.0 -1.0; -1.0 9.0 ])), output = MvNormalMeanPrecision([ -1.0, 2.0 ], [ 7.0 -1.0; -1.0 9.0 ])),
            (input = (m_out = PointMass([ 0.0, 0.0 ]),  m_Λ = PointMass([ 1.0 0.0; 0.0 1.0 ])),   output = MvNormalMeanPrecision([ 0.0, 0.0 ], [ 1.0 0.0; 0.0 1.0 ]))
        ]

    end

    @testset "Belief Propagation: (m_out::MultivariateNormalDistributionsFamily, m_Λ::PointMass)" begin

        @test_rules [ with_float_conversions = true ] MvNormalMeanPrecision(:μ, Marginalisation) [
            (input = (m_out = MvNormalMeanPrecision([ 2.0, 1.0 ], [ 3.0 2.0; 2.0 4.0 ]),   m_Λ = PointMass([ 6.0 4.0; 4.0 8.0 ])),    output = MvNormalMeanCovariance([ 2.0, 1.0 ], [ 0.75 -0.375; -0.375 0.5625 ])),
            (input = (m_out = MvNormalMeanPrecision([ 0.0, 0.0 ], [ 7.0 -1.0; -1.0 9.0 ]), m_Λ = PointMass([ 12.0 -2.0; -2.0 7.0 ])), output = MvNormalMeanCovariance([ 0.0, 0.0 ], [ 577/2480 51/1240; 51/1240 163/620 ])),
            (input = (m_out = MvNormalMeanPrecision([ 3.0, -1.0 ], [ 1.0 0.0; 0.0 1.0 ]),  m_Λ = PointMass([ 1.0 0.0; 0.0 1.0 ])),    output = MvNormalMeanCovariance([ 3.0, -1.0 ], [ 2.0 0.0; 0.0 2.0 ])),
        ]

        @test_rules [ with_float_conversions = true ] MvNormalMeanPrecision(:μ, Marginalisation) [
            (input = (m_out = MvNormalMeanCovariance([ 2.0, 1.0 ], [ 3.0 2.0; 2.0 4.0 ]),  m_Λ = PointMass([ 6.0 4.0; 4.0 8.0 ])),     output = MvNormalMeanCovariance([ 2.0, 1.0 ], [ 13/4 15/8; 15/8 67/16])),
            (input = (m_out = MvNormalMeanCovariance([ 0.0, 0.0 ], [ 7.0 -1.0; -1.0 9.0 ]), m_Λ = PointMass([ 12.0 -2.0; -2.0 7.0 ])), output = MvNormalMeanCovariance([ 0.0, 0.0 ], [ 567/80 -39/40; -39/40 183/20 ])),
            (input = (m_out = MvNormalMeanCovariance([ 3.0, -1.0 ], [ 1.0 0.0; 0.0 1.0 ]),  m_Λ = PointMass([ 1.0 0.0; 0.0 1.0 ])),    output = MvNormalMeanCovariance([ 3.0, -1.0 ], [ 2.0 0.0; 0.0 2.0 ]))
        ]

        @test_rules [ with_float_conversions = true ] MvNormalMeanPrecision(:μ, Marginalisation) [
            (input = (m_out = MvNormalWeightedMeanPrecision([ 2.0, 1.0 ], [ 3.0 2.0; 2.0 4.0 ]),   m_Λ = PointMass([ 6.0 4.0; 4.0 8.0 ])),    output = MvNormalMeanCovariance([ 3/4, -1/8 ], [ 0.75 -0.375; -0.375 0.5625 ])),
            (input = (m_out = MvNormalWeightedMeanPrecision([ 0.0, 0.0 ], [ 7.0 -1.0; -1.0 9.0 ]), m_Λ = PointMass([ 12.0 -2.0; -2.0 7.0 ])), output = MvNormalMeanCovariance([ 0.0, 0.0 ], [ 577/2480 51/1240; 51/1240 163/620 ])),
            (input = (m_out = MvNormalWeightedMeanPrecision([ 3.0, -1.0 ], [ 1.0 0.0; 0.0 1.0 ]),  m_Λ = PointMass([ 1.0 0.0; 0.0 1.0 ])),    output = MvNormalMeanCovariance([ 3.0, -1.0 ], [ 2.0 0.0; 0.0 2.0 ])),
        ]

    end

    @testset "Variational: (q_out::Any, q_Λ::Any)" begin

        @test_rules [ with_float_conversions = true ] MvNormalMeanPrecision(:μ, Marginalisation) [
            (input = (q_out = PointMass([ -1.0 ]), q_Λ = PointMass([ 2.0 ])), output = MvNormalMeanPrecision([ -1.0 ], [ 2.0 ])),
            (input = (q_out = PointMass([  1.0 ]), q_Λ = PointMass([ 2.0 ])), output = MvNormalMeanPrecision([  1.0 ], [ 2.0 ])),
            (input = (q_out = PointMass([ 2.0 ]),  q_Λ = PointMass([ 1.0 ])), output = MvNormalMeanPrecision([  2.0 ], [ 1.0 ])),
            (input = (q_out = PointMass([ 1.0, 3.0 ]),  q_Λ = PointMass([ 3.0 2.0; 2.0 4.0 ])),   output = MvNormalMeanPrecision([ 1.0, 3.0 ], [ 3.0 2.0; 2.0 4.0 ])),
            (input = (q_out = PointMass([ -1.0, 2.0 ]), q_Λ = PointMass([ 7.0 -1.0; -1.0 9.0 ])), output = MvNormalMeanPrecision([ -1.0, 2.0 ], [ 7.0 -1.0; -1.0 9.0 ])),
            (input = (q_out = PointMass([ 0.0, 0.0 ]),  q_Λ = PointMass([ 1.0 0.0; 0.0 1.0 ])),   output = MvNormalMeanPrecision([ 0.0, 0.0 ], [ 1.0 0.0; 0.0 1.0 ]))
        ]

        @test_rules [ with_float_conversions = true ] MvNormalMeanPrecision(:μ, Marginalisation) [
            (input = (q_out = MvNormalMeanCovariance([ 2.0, 1.0 ], [ 3.0 2.0; 2.0 4.0 ]), q_Λ = PointMass([ 6.0 4.0; 4.0 8.0 ])), output = MvNormalMeanPrecision([ 2.0, 1.0 ], [ 6.0 4.0; 4.0 8.0 ])),
            (input = (q_out = MvNormalMeanPrecision([ 2.0, 1.0 ], [ 3.0 2.0; 2.0 4.0 ]), q_Λ = PointMass([ 6.0 4.0; 4.0 8.0 ])), output = MvNormalMeanPrecision([ 2.0, 1.0 ], [ 6.0 4.0; 4.0 8.0 ])),
            (input = (q_out = MvNormalWeightedMeanPrecision([ 2.0, 1.0 ], [ 3.0 2.0; 2.0 4.0 ]), q_Λ = PointMass([ 6.0 4.0; 4.0 8.0 ])), output = MvNormalMeanPrecision([ 3/4, -1/8 ], [ 6.0 4.0; 4.0 8.0 ])),
        ]

    end

    @testset "Structured variational: (m_out::PointMass, q_Σ::Any)" begin
        
        @test_rules [ with_float_conversions = true ] MvNormalMeanPrecision(:μ, Marginalisation) [
            (input = (m_out = PointMass([ 1.0, 3.0 ]),  q_Λ = PointMass([ 3.0 2.0; 2.0 4.0 ])),   output = MvNormalMeanPrecision([ 1.0, 3.0 ], [ 3.0 2.0; 2.0 4.0 ])),
            (input = (m_out = PointMass([ -1.0, 2.0 ]), q_Λ = PointMass([ 7.0 -1.0; -1.0 9.0 ])), output = MvNormalMeanPrecision([ -1.0, 2.0 ], [ 7.0 -1.0; -1.0 9.0 ])),
            (input = (m_out = PointMass([ 0.0, 0.0 ]),  q_Λ = PointMass([ 1.0 0.0; 0.0 1.0 ])),   output = MvNormalMeanPrecision([ 0.0, 0.0 ], [ 1.0 0.0; 0.0 1.0 ])),
        ]
        
    end

    @testset "Structured variational: (m_out::MvNormalMeanPrecision, q_Λ::Any)" begin
        
        @test_rules [ with_float_conversions = true ] MvNormalMeanPrecision(:μ, Marginalisation) [
            (input = (m_out = MvNormalMeanPrecision([ 2.0, 1.0 ], [ 3.0 2.0; 2.0 4.0 ]),   q_Λ = PointMass([ 6.0 4.0; 4.0 8.0 ])),    output = MvNormalMeanCovariance([ 2.0, 1.0 ], [ 0.75 -0.375; -0.375 0.5625 ])),
            (input = (m_out = MvNormalMeanPrecision([ 0.0, 0.0 ], [ 7.0 -1.0; -1.0 9.0 ]), q_Λ = PointMass([ 12.0 -2.0; -2.0 7.0 ])), output = MvNormalMeanCovariance([ 0.0, 0.0 ], [ 577/2480 51/1240; 51/1240 163/620 ])),
            (input = (m_out = MvNormalMeanPrecision([ 3.0, -1.0 ], [ 1.0 0.0; 0.0 1.0 ]),  q_Λ = PointMass([ 1.0 0.0; 0.0 1.0 ])),    output = MvNormalMeanCovariance([ 3.0, -1.0 ], [ 2.0 0.0; 0.0 2.0 ])),
        ]

        @test_rules [ with_float_conversions = true ] MvNormalMeanPrecision(:μ, Marginalisation) [
            (input = (m_out = MvNormalMeanCovariance([ 2.0, 1.0 ], [ 3.0 2.0; 2.0 4.0 ]),  q_Λ = PointMass([ 6.0 4.0; 4.0 8.0 ])),     output = MvNormalMeanCovariance([ 2.0, 1.0 ], [ 13/4 15/8; 15/8 67/16])),
            (input = (m_out = MvNormalMeanCovariance([ 0.0, 0.0 ], [ 7.0 -1.0; -1.0 9.0 ]), q_Λ = PointMass([ 12.0 -2.0; -2.0 7.0 ])), output = MvNormalMeanCovariance([ 0.0, 0.0 ], [ 567/80 -39/40; -39/40 183/20 ])),
            (input = (m_out = MvNormalMeanCovariance([ 3.0, -1.0 ], [ 1.0 0.0; 0.0 1.0 ]),  q_Λ = PointMass([ 1.0 0.0; 0.0 1.0 ])),    output = MvNormalMeanCovariance([ 3.0, -1.0 ], [ 2.0 0.0; 0.0 2.0 ]))
        ]
        
        @test_rules [ with_float_conversions = true ] MvNormalMeanPrecision(:μ, Marginalisation) [
            (input = (m_out = MvNormalWeightedMeanPrecision([ 2.0, 1.0 ], [ 3.0 2.0; 2.0 4.0 ]),   q_Λ = PointMass([ 6.0 4.0; 4.0 8.0 ])),    output = MvNormalMeanCovariance([ 3/4, -1/8 ], [ 0.75 -0.375; -0.375 0.5625 ])),
            (input = (m_out = MvNormalWeightedMeanPrecision([ 0.0, 0.0 ], [ 7.0 -1.0; -1.0 9.0 ]), q_Λ = PointMass([ 12.0 -2.0; -2.0 7.0 ])), output = MvNormalMeanCovariance([ 0.0, 0.0 ], [ 577/2480 51/1240; 51/1240 163/620 ])),
            (input = (m_out = MvNormalWeightedMeanPrecision([ 3.0, -1.0 ], [ 1.0 0.0; 0.0 1.0 ]),  q_Λ = PointMass([ 1.0 0.0; 0.0 1.0 ])),    output = MvNormalMeanCovariance([ 3.0, -1.0 ], [ 2.0 0.0; 0.0 2.0 ])),
        ]

        @test_rules [ with_float_conversions = true ] MvNormalMeanPrecision(:μ, Marginalisation) [
            (input = (m_out = MvNormalMeanPrecision([ 2.0, 1.0 ], [ 3.0 2.0; 2.0 4.0 ]),   q_Λ = Wishart(2.0, [ 6.0 4.0; 4.0 8.0 ] ./ 2.0)),    output = MvNormalMeanCovariance([ 2.0, 1.0 ], [ 0.75 -0.375; -0.375 0.5625 ])),
            (input = (m_out = MvNormalMeanPrecision([ 0.0, 0.0 ], [ 7.0 -1.0; -1.0 9.0 ]), q_Λ = Wishart(3.0, [ 12.0 -2.0; -2.0 7.0 ] ./ 3.0)), output = MvNormalMeanCovariance([ 0.0, 0.0 ], [ 577/2480 51/1240; 51/1240 163/620 ])),
            (input = (m_out = MvNormalMeanPrecision([ 3.0, -1.0 ], [ 1.0 0.0; 0.0 1.0 ]),  q_Λ = Wishart(4.0, [ 1.0 0.0; 0.0 1.0 ] ./ 4.0)),    output = MvNormalMeanCovariance([ 3.0, -1.0 ], [ 2.0 0.0; 0.0 2.0 ])),
        ]

        @test_rules [ with_float_conversions = true ] MvNormalMeanPrecision(:μ, Marginalisation) [
            (input = (m_out = MvNormalMeanCovariance([ 2.0, 1.0 ], [ 3.0 2.0; 2.0 4.0 ]),  q_Λ = Wishart(2.0, [ 6.0 4.0; 4.0 8.0 ] ./ 2.0)),     output = MvNormalMeanCovariance([ 2.0, 1.0 ], [ 13/4 15/8; 15/8 67/16])),
            (input = (m_out = MvNormalMeanCovariance([ 0.0, 0.0 ], [ 7.0 -1.0; -1.0 9.0 ]), q_Λ = Wishart(3.0, [ 12.0 -2.0; -2.0 7.0 ] ./ 3.0)), output = MvNormalMeanCovariance([ 0.0, 0.0 ], [ 567/80 -39/40; -39/40 183/20 ])),
            (input = (m_out = MvNormalMeanCovariance([ 3.0, -1.0 ], [ 1.0 0.0; 0.0 1.0 ]),  q_Λ = Wishart(4.0, [ 1.0 0.0; 0.0 1.0 ] ./ 4.0)),    output = MvNormalMeanCovariance([ 3.0, -1.0 ], [ 2.0 0.0; 0.0 2.0 ]))
        ]
        
        @test_rules [ with_float_conversions = true ] MvNormalMeanPrecision(:μ, Marginalisation) [
            (input = (m_out = MvNormalWeightedMeanPrecision([ 2.0, 1.0 ], [ 3.0 2.0; 2.0 4.0 ]),   q_Λ = Wishart(2.0, [ 6.0 4.0; 4.0 8.0 ] ./ 2.0)),    output = MvNormalMeanCovariance([ 3/4, -1/8 ], [ 0.75 -0.375; -0.375 0.5625 ])),
            (input = (m_out = MvNormalWeightedMeanPrecision([ 0.0, 0.0 ], [ 7.0 -1.0; -1.0 9.0 ]), q_Λ = Wishart(3.0, [ 12.0 -2.0; -2.0 7.0 ] ./ 3.0)), output = MvNormalMeanCovariance([ 0.0, 0.0 ], [ 577/2480 51/1240; 51/1240 163/620 ])),
            (input = (m_out = MvNormalWeightedMeanPrecision([ 3.0, -1.0 ], [ 1.0 0.0; 0.0 1.0 ]),  q_Λ = Wishart(4.0, [ 1.0 0.0; 0.0 1.0 ] ./ 4.0)),    output = MvNormalMeanCovariance([ 3.0, -1.0 ], [ 2.0 0.0; 0.0 2.0 ])),
        ]
        
    end

end



end