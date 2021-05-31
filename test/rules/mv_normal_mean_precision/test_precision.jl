module RulesMvNormalMeanPrecisionPrecisionTest

using Test
using ReactiveMP
using Random


import ReactiveMP: @test_rules

@testset "rules:MvNormalMeanPrecision:precision" begin

    @testset "Variational: (q_out::PointMass, q_μ::MultivariateNormalDistributionsFamily)" begin

        @test_rules [ with_float_conversions = false ] MvNormalMeanPrecision(:Λ, Marginalisation) [
            (input = (q_out = PointMass([ 1.0, 2.0 ]), q_μ = MvNormalMeanPrecision([ 3.0, 5.0 ], [ 3.0 2.0; 2.0 4.0 ])), output = Wishart(4.0, [ 75/73 -46/73; -46/73 36/73 ])),
            (input = (q_out = MvNormalMeanPrecision([1.0; 2.0], [3.0 2.0; 2.0 4.0]), q_μ = PointMass([3.0; 5.0])), output = Wishart(4.0, [ 75/73 -46/73; -46/73 36/73 ])),
            (input = (q_out = MvNormalMeanPrecision([1.0; 2.0], [3.0 2.0; 2.0 4.0]), q_μ = MvNormalMeanPrecision([3.0; 5.0], [3.0 2.0; 2.0 4.0])), output = Wishart(4.0, [ 39/74 -11/37; -11/37 10/37 ])),
            (input = (q_out = MvNormalMeanPrecision([1.0; 2.0; 3.0], [3.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 4.0]), q_μ = MvNormalMeanPrecision([3.0; 5.0; -2.0], [3.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 4.0])), output = Wishart(5.0, [ 15/11 -3/22 5/11; -3/22 19/22 5/11; 5/11 5/11 16/33 ]))
        ]
        @test_rules [ with_float_conversions = false ] MvNormalMeanPrecision(:Λ, Marginalisation) [
            (input = (q_out = PointMass([ 1.0, 2.0 ]), q_μ = MvNormalMeanCovariance([ 3.0, 5.0 ], [ 3.0 2.0; 2.0 4.0 ])), output = Wishart(4.0, [ 13/27 -8/27; -8/27 7/27 ])),
            (input = (q_out = MvNormalMeanCovariance([1.0; 2.0], [3.0 2.0; 2.0 4.0]), q_μ = PointMass([3.0; 5.0])), output = Wishart(4.0, [ 13/27 -8/27; -8/27 7/27 ])),
            (input = (q_out = MvNormalMeanCovariance([1.0; 2.0], [3.0 2.0; 2.0 4.0]), q_μ = MvNormalMeanCovariance([3.0; 5.0], [3.0 2.0; 2.0 4.0])), output = Wishart(4.0, [ 17/70 -1/7; -1/7 1/7 ])),
            (input = (q_out = MvNormalMeanCovariance([1.0; 2.0; 3.0], [3.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 4.0]), q_μ = MvNormalMeanCovariance([3.0; 5.0; -2.0], [3.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 4.0])), output = Wishart(5.0, [ 51/338 -6/169 5/169; -6/169 115/676 45/676; 5/169 45/676 47/676 ]))
        ]
        @test_rules [ with_float_conversions = false ] MvNormalMeanPrecision(:Λ, Marginalisation) [
            (input = (q_out = PointMass([ 1.0, 2.0 ]), q_μ = MvNormalWeightedMeanPrecision([ 3.0, 5.0 ], [ 3.0 2.0; 2.0 4.0 ])), output = Wishart(4.0, [ 73/67 -26/67; -26/67 68/67 ])),
            (input = (q_out = MvNormalWeightedMeanPrecision([1.0; 2.0], [3.0 2.0; 2.0 4.0]), q_μ = PointMass([3.0; 5.0])), output = Wishart(4.0, [ 165/163 -106/163; -106/163 76/163 ])),
            (input = (q_out = MvNormalWeightedMeanPrecision([1.0; 2.0], [3.0 2.0; 2.0 4.0]), q_μ = MvNormalWeightedMeanPrecision([3.0; 5.0], [3.0 2.0; 2.0 4.0])), output = Wishart(4.0, [ 73/70 11/35; 11/35 34/35 ])),
            (input = (q_out = MvNormalWeightedMeanPrecision([1.0; 2.0; 3.0], [3.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 4.0]), q_μ = MvNormalWeightedMeanPrecision([3.0; 5.0; -2.0], [3.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 4.0])), output = Wishart(5.0, [ 459/338 -36/169 60/169; -36/169 115/169 90/169; 60/169 90/169 188/169 ]))
        ]

    end

end



end