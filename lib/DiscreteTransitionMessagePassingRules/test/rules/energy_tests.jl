# The node's type, interfaces and factorisation come from its declaration. The fourth interface
# is the group `T`, whose members are `(:T, k)`.
@testitem "rules:DiscreteTransition:node properties" tags = [:rules] begin
    using DiscreteTransitionMessagePassingRules, MessagePassingRulesBase

    spec = MessagePassingRulesBase.nodespec(DiscreteTransition)
    @test spec.type == Stochastic()
    @test map(i -> i.name, spec.interfaces) == (:out, :in, :a, :T)
    @test map(i -> i.group, spec.interfaces) == (false, false, false, true)
    @test spec.min_group_length == 0
end

@testitem "rules:DiscreteTransition:energy:(q_out_in::Contingency, q_a::PointMass)" tags = [:rules] begin
    using DiscreteTransitionMessagePassingRules, MessagePassingRulesBase, BayesBase, ExponentialFamily, Distributions, LinearAlgebra
    using BayesBase: tiny

    diageye(n) = Matrix{Float64}(I, n, n)
    energy(q_out_in, q_a) = getresult(call_average_energy(DiscreteTransition; clusters = ((:out, :in) => q_out_in,), q = (a = q_a,)))

    contingency_matrix = [0.2 0.3; 0.4 0.1]
    a_matrix = [0.7 0.3; 0.2 0.8]
    # Expected value calculated by hand
    expected = -sum(contingency_matrix .* log.(clamp.(a_matrix, tiny, Inf)))
    @test energy(Contingency(contingency_matrix), PointMass(a_matrix)) ≈ expected

    contingency_matrix = [0.2 0.3; 0.4 0.1]
    a_matrix = [1.0 0.0; 0.0 1.0]
    expected = -sum(contingency_matrix .* log.(clamp.(a_matrix, tiny, Inf)))
    @test energy(Contingency(contingency_matrix), PointMass(a_matrix)) ≈ expected

    contingency_matrix = prod.(Iterators.product([0, 1, 0], [0.1, 0.4, 0.5]))
    a_matrix = diageye(3)
    expected = -sum(contingency_matrix .* log.(clamp.(a_matrix, tiny, Inf)))
    @test energy(Contingency(contingency_matrix), PointMass(a_matrix)) ≈ expected

    contingency_matrix = [0.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 0.0]
    expected = -sum(contingency_matrix .* log.(clamp.(a_matrix, tiny, Inf)))
    @test energy(Contingency(contingency_matrix), PointMass(diageye(3))) ≈ expected
end

@testitem "rules:DiscreteTransition:energy:(q_out::Any, q_in::Any, q_a::PointMass)" tags = [:rules] begin
    using DiscreteTransitionMessagePassingRules, MessagePassingRulesBase, BayesBase, ExponentialFamily, Distributions
    using BayesBase: tiny

    energy(q_out, q_in, q_a) = getresult(call_average_energy(DiscreteTransition; q = (out = q_out, in = q_in, a = q_a)))

    q_out = Categorical([0.3, 0.7])
    q_in = Categorical([0.8, 0.2])
    q_a = PointMass([0.7 0.3; 0.2 0.8])
    contingency = probvec(q_out) * probvec(q_in)'
    expected = -sum(contingency .* log.(clamp.(mean(q_a), tiny, Inf)))
    @test energy(q_out, q_in, q_a) ≈ expected

    q_out = Categorical([0.0, 1.0])
    q_in = Categorical([0.0, 1.0])
    q_a = PointMass([1.0 0.0; 1.0 0.0])
    contingency = probvec(q_out) * probvec(q_in)'
    expected = -sum(contingency .* log.(clamp.(mean(q_a), tiny, Inf)))
    @test energy(q_out, q_in, q_a) ≈ expected
end

@testitem "rules:DiscreteTransition:energy:(q_out_in::Contingency, q_T1_T2::Contingency, q_a::Any)" tags = [:rules] begin
    using DiscreteTransitionMessagePassingRules, MessagePassingRulesBase, BayesBase, ExponentialFamily, Distributions
    using BayesBase: clamplog
    import Base.Broadcast: BroadcastFunction

    # The joint over both `T`s covers the whole group, and is keyed by it.
    energy(q_out_in, q_T1_T2, q_a) = getresult(call_average_energy(DiscreteTransition; clusters = ((:out, :in) => q_out_in, (:T,) => q_T1_T2), q = (a = q_a,)))

    q_out_in = Contingency([0.3 0.7; 0.4 0.6])
    q_T1_T2 = Contingency([0.8 0.2; 0.1 0.9])
    q_a = DirichletCollection(
        [
            3.0 4.0; 8.0 5.0;;; 9.0 10.0; 6.0 3.0;;;;
            1.0 4.0; 8.0 9.0;;; 9.0 10.0; 1.0 2.0
        ],
    )
    contingency = reshape(components(q_out_in), 2, 2, 1, 1) .* reshape(components(q_T1_T2), 1, 1, 2, 2)
    expected = -sum(contingency .* mean(BroadcastFunction(clamplog), q_a))
    @test energy(q_out_in, q_T1_T2, q_a) ≈ expected

    q_a = PointMass(
        [
            0.29693261210360755 0.48331963608086737; 0.7030673878963924 0.5166803639191326;;; 0.6678183242774415 0.827095096579412; 0.3321816757225585 0.17290490342058795;;;;
            0.11877772619163436 0.3941346676252447; 0.8812222738083656 0.6058653323747553;;; 0.6876122374136755 0.9388959627009439; 0.3123877625863246 0.06110403729905605
        ],
    )
    contingency = reshape(components(q_out_in), 2, 2, 1, 1) .* reshape(components(q_T1_T2), 1, 1, 2, 2)
    expected = -sum(contingency .* mean(BroadcastFunction(clamplog), q_a))
    @test energy(q_out_in, q_T1_T2, q_a) ≈ expected
end
