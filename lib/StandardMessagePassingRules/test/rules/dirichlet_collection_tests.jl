# DirichletCollection: the cases are hand-derived, and the energy is
# checked against the Dirichlet node's, column by column, for rank 2 (DiscreteTransition's
# case) and rank 3.

@testitem "rules:DirichletCollection:out-marginals" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, ExponentialFamily, BayesBase, Distributions

    a = [1.0 2.0; 3.0 0.5; 2.0 1.5]
    @test_message_update_rule(
        node = DirichletCollection, target = :out,
        cases = [
            (m = (a = PointMass(a),),) => DirichletCollection(a),
            (q = (a = PointMass(a),),) => DirichletCollection(a),
        ],
    )
    # The product of two collections adds their parameters, less one: a + α - 1.
    α = [2.0 1.0; 1.5 3.0; 1.0 2.5]
    @test_marginal_update_rule(
        node = DirichletCollection, target = (:out, :a),
        cases = [(m = (out = DirichletCollection(α), a = PointMass(a)),) => FactorizedCluster((:out,) => DirichletCollection(a .+ α .- 1), (:a,) => PointMass(a))],
    )
end

@testitem "rules:DirichletCollection:energy" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, ExponentialFamily, BayesBase, Distributions

    columns(x) = eachslice(x, dims = Tuple(2:ndims(x)))
    by_columns(α, a) = sum(getresult(call_average_energy(Dirichlet; q = (out = Dirichlet(collect(α_k)), a = PointMass(collect(a_k))))) for (α_k, a_k) in zip(columns(α), columns(a)))

    α2, a2 = [2.0 1.0; 1.5 3.0; 1.0 2.5], [1.0 2.0; 3.0 0.5; 2.0 1.5]
    α3, a3 = reshape(collect(1.0:12.0) ./ 4, 3, 2, 2), reshape(collect(12.0:-1.0:1.0) ./ 5, 3, 2, 2)
    @test_average_energy(
        node = DirichletCollection,
        cases = [
            (q = (out = DirichletCollection(α2), a = PointMass(a2)),) => by_columns(α2, a2),
            (q = (out = DirichletCollection(α3), a = PointMass(a3)),) => by_columns(α3, a3),
        ],
    )
end
