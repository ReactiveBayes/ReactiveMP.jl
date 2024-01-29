module NodesNormalTest

using ReactiveMP, Random, BayesBase, ExponentialFamily, Distributions

import ReactiveMP: make_node, FactorNodeCreationOptions

@testitem "NormalNode" begin
    out = randomvar(:out)
    mean = randomvar(:mean)
    std = randomvar(:std)

    # We do not allow creation of the `Normal` node (as for now, can be changed in the future)
    # A user should use a more specific version instead, e.g. `NormalMeanVariance`
    @test_throws ErrorException make_node(Normal, FactorNodeCreationOptions(), out, mean, std)

    # We do not allow creation of the `MvNormal` node (as for now, can be changed in the future)
    # A user should use a more specific version instead, e.g. `MvNormalMeanCovariance`
    @test_throws ErrorException make_node(MvNormal, FactorNodeCreationOptions(), out, mean, std)
end

end
