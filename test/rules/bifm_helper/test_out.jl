module RulesBIFMHelperOutTest

using Test
using ReactiveMP
using Random
import ReactiveMP: @test_rules

@testset "rules:BIFMHelper:out" begin
    @testset "Variational Message Passing: (q_in::Any, )" begin
        @test_rules [with_float_conversions = true] BIFMHelper(:out, Marginalisation) [
            (
                input = (q_in = MvNormalMeanCovariance([1.0, 2.0], [3.0 0; 0 2.0]),),
                output = ProdFinal(MvNormalMeanCovariance([1.0, 2.0], [3.0 0; 0 2.0]))
            ),
            (input = (q_in = NormalMeanVariance(9.5, 3.2),), output = ProdFinal(NormalMeanVariance(9.5, 3.2))),
            (
                input = (q_in = MvNormalWeightedMeanPrecision([1.0, 2.0], [3.0 0; 0 2.0]),),
                output = ProdFinal(MvNormalWeightedMeanPrecision([1.0, 2.0], [3.0 0; 0 2.0]))
            )
        ]
    end
end

end
