
@testitem "rules:BIFMHelper:out" begin
    using ReactiveMP, Random, ExponentialFamily, BayesBase

    import ReactiveMP: @test_rules

    @testset "Variational Message Passing: (q_in::Any, )" begin
        @test_rules [check_type_promotion = true] BIFMHelper(:out, Marginalisation) [
            (input = (q_in = MvNormalMeanCovariance([1.0, 2.0], [3.0 0; 0 2.0]),), output = TerminalProdArgument(MvNormalMeanCovariance([1.0, 2.0], [3.0 0; 0 2.0]))),
            (input = (q_in = NormalMeanVariance(9.5, 3.2),), output = TerminalProdArgument(NormalMeanVariance(9.5, 3.2))),
            (input = (q_in = MvNormalWeightedMeanPrecision([1.0, 2.0], [3.0 0; 0 2.0]),), output = TerminalProdArgument(MvNormalWeightedMeanPrecision([1.0, 2.0], [3.0 0; 0 2.0])))
        ]
    end
end