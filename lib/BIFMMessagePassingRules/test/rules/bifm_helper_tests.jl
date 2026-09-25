@testitem "rules:BIFMHelper:in" tags = [:rules] begin
    using BIFMMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, BayesBase, ExponentialFamily, Distributions

    @test_message_update_rule(
        node = BIFMHelper, target = :in,
        cases = [
            (m = (out = PointMass(1.0),),) => PointMass(1.0),
            (m = (out = NormalMeanVariance(9.5, 3.2),),) => NormalMeanVariance(9.5, 3.2),
            (m = (out = MvNormalWeightedMeanPrecision([1.0, 2.0], [3.0 0; 0 2.0]),),) => MvNormalWeightedMeanPrecision([1.0, 2.0], [3.0 0; 0 2.0]),
        ],
    )
end

@testitem "rules:BIFMHelper:out" tags = [:rules] begin
    using BIFMMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, BayesBase, ExponentialFamily, Distributions

    @test_message_update_rule(
        node = BIFMHelper, target = :out,
        cases = [
            (q = (in = MvNormalMeanCovariance([1.0, 2.0], [3.0 0; 0 2.0]),),) => TerminalProdArgument(MvNormalMeanCovariance([1.0, 2.0], [3.0 0; 0 2.0])),
            (q = (in = NormalMeanVariance(9.5, 3.2),),) => TerminalProdArgument(NormalMeanVariance(9.5, 3.2)),
            (q = (in = MvNormalWeightedMeanPrecision([1.0, 2.0], [3.0 0; 0 2.0]),),) => TerminalProdArgument(MvNormalWeightedMeanPrecision([1.0, 2.0], [3.0 0; 0 2.0])),
        ],
    )
end

# The energy throws, since the free energy of a BIFM model is not supported.
@testitem "rules:BIFMHelper:energy" tags = [:rules] begin
    using BIFMMessagePassingRules, MessagePassingRulesBase, BayesBase, ExponentialFamily, Distributions

    q = (out = MvNormalMeanCovariance([1.0, 1.0], [2.0 0; 0 3.0]), in = MvNormalMeanCovariance([1.0, 1.0], [2.0 0; 0 3.0]))
    @test_throws BIFMMessagePassingRules.BIFMFreeEnergyError getresult(call_average_energy(BIFMHelper; q))
    error = try
        getresult(call_average_energy(BIFMHelper; q = (out = MvNormalMeanCovariance([1.0, 2.0], [2.0 0; 0 1.0]), in = MvNormalMeanPrecision([1.0, 2.0], [0.5 0; 0 1.0]))))
    catch e
        e
    end
    @test error isa BIFMMessagePassingRules.BIFMFreeEnergyError
    @test error.node === :BIFMHelper
    @test occursin("`BIFMHelper` is not supported", sprint(showerror, error))
end
