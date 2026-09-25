@testitem "rules:multivariate-normals:energy-float-type" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, ExponentialFamily, BayesBase, Distributions

    # Every multivariate normal's energy stays in its inputs' float type: d log 2π is computed
    # in it, not in Float64.
    I2 = Float32[1 0; 0 1]
    x, y = MvNormalMeanCovariance(Float32[1, 2], I2), MvNormalMeanCovariance(Float32[0, 1], 2 * I2)
    joint = MvNormalMeanCovariance(Float32[1, 2, 0, 1], Float32[1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1])
    W, γ = Wishart(3.0f0, I2), GammaShapeRate(2.0f0, 1.0f0)
    cases = [
        (MvNormalMeanCovariance, (q = (out = x, μ = y, Σ = PointMass(I2)),)),
        (MvNormalMeanCovariance, (q = (Σ = PointMass(I2),), clusters = ((:out, :μ) => joint,))),
        (MvNormalMeanPrecision, (q = (out = x, μ = y, Λ = W),)),
        (MvNormalMeanPrecision, (q = (out = x, μ = y, Λ = PointMass(I2)),)),
        (MvNormalMeanPrecision, (q = (Λ = W,), clusters = ((:out, :μ) => joint,))),
        (MvNormalWeightedMeanPrecision, (q = (out = x, ξ = y, Λ = PointMass(I2)),)),
        (MvNormalMeanScalePrecision, (q = (out = x, μ = y, γ = γ),)),
        (MvNormalMeanScalePrecision, (q = (γ = γ,), clusters = ((:out, :μ) => joint,))),
        (MvNormalMeanScaleMatrixPrecision, (q = (out = x, μ = y, γ = γ, G = W),)),
        (MvNormalMeanScaleMatrixPrecision, (q = (γ = γ, G = W), clusters = ((:out, :μ) => joint,))),
    ]
    for (node, inputs) in cases
        @test getresult(call_average_energy(node; inputs...)) isa Float32
    end
    # ExponentialFamily 2.6's E[log |Σ|] for an InverseWishart computes `d * log(2)`, a Float64
    # (ExponentialFamily.jl#322).
    @test_broken getresult(call_average_energy(MvNormalMeanCovariance; q = (out = x, μ = y, Σ = InverseWishart(4.0f0, I2)))) isa Float32
end
