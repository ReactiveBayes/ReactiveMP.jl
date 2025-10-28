
@testitem "NormalMixtureNode" begin
    using ReactiveMP, Random, BayesBase, ExponentialFamily

    import ReactiveMP: ManyOf
    import ExponentialFamily: WishartFast

    @testset "AverageEnergy" begin
        begin
            q_out = NormalMeanVariance(0.0, 1.0)
            q_switch = Bernoulli(0.2)
            q_m = (NormalMeanVariance(1.0, 2.0), NormalMeanVariance(3.0, 4.0))
            q_p = (GammaShapeRate(2.0, 3.0), GammaShapeRate(4.0, 5.0))

            marginals = (
                Marginal(q_out, false, false, nothing),
                Marginal(q_switch, false, false, nothing),
                ManyOf(map(q_m_ -> Marginal(q_m_, false, false, nothing), q_m)),
                ManyOf(map(q_p_ -> Marginal(q_p_, false, false, nothing), q_p))
            )

            ref_val =
                0.8 * (score(AverageEnergy(), NormalMeanPrecision, Val{(:out, :μ, :τ)}(), map((q) -> Marginal(q, false, false, nothing), (q_out, q_m[1], q_p[1])), nothing)) +
                0.2 * (score(AverageEnergy(), NormalMeanPrecision, Val{(:out, :μ, :τ)}(), map((q) -> Marginal(q, false, false, nothing), (q_out, q_m[2], q_p[2])), nothing))
            @test score(AverageEnergy(), NormalMixture, Val{(:out, :switch, :m, :p)}(), marginals, nothing) ≈ ref_val
        end

        begin
            q_out = NormalMeanVariance(1.0, 1.0)
            q_switch = Bernoulli(0.4)
            q_m = (NormalMeanVariance(3.0, 2.0), NormalMeanVariance(3.0, 4.0))
            q_p = (GammaShapeRate(2.0, 3.0), GammaShapeRate(1.0, 5.0))

            marginals = (
                Marginal(q_out, false, false, nothing),
                Marginal(q_switch, false, false, nothing),
                ManyOf(map(q_m_ -> Marginal(q_m_, false, false, nothing), q_m)),
                ManyOf(map(q_p_ -> Marginal(q_p_, false, false, nothing), q_p))
            )

            ref_val =
                0.6 * (score(AverageEnergy(), NormalMeanPrecision, Val{(:out, :μ, :τ)}(), map((q) -> Marginal(q, false, false, nothing), (q_out, q_m[1], q_p[1])), nothing)) +
                0.4 * (score(AverageEnergy(), NormalMeanPrecision, Val{(:out, :μ, :τ)}(), map((q) -> Marginal(q, false, false, nothing), (q_out, q_m[2], q_p[2])), nothing))
            @test score(AverageEnergy(), NormalMixture, Val{(:out, :switch, :m, :p)}(), marginals, nothing) ≈ ref_val
        end

        begin
            q_out = NormalMeanVariance(0.0, 1.0)
            q_switch = Categorical([0.5, 0.5])
            q_m = (NormalMeanPrecision(1.0, 2.0), NormalMeanPrecision(3.0, 4.0))
            q_p = (GammaShapeRate(3.0, 3.0), GammaShapeRate(4.0, 5.0))

            marginals = (
                Marginal(q_out, false, false, nothing),
                Marginal(q_switch, false, false, nothing),
                ManyOf(map(q_m_ -> Marginal(q_m_, false, false, nothing), q_m)),
                ManyOf(map(q_p_ -> Marginal(q_p_, false, false, nothing), q_p))
            )

            ref_val =
                0.5 * (score(AverageEnergy(), NormalMeanPrecision, Val{(:out, :μ, :τ)}(), map((q) -> Marginal(q, false, false, nothing), (q_out, q_m[1], q_p[1])), nothing)) +
                0.5 * (score(AverageEnergy(), NormalMeanPrecision, Val{(:out, :μ, :τ)}(), map((q) -> Marginal(q, false, false, nothing), (q_out, q_m[2], q_p[2])), nothing))
            @test score(AverageEnergy(), NormalMixture, Val{(:out, :switch, :m, :p)}(), marginals, nothing) ≈ ref_val
        end

        begin
            q_out = MvNormalMeanCovariance([0.0], [1.0])
            q_switch = Categorical([0.5, 0.5])
            q_m = (MvNormalMeanPrecision([1.0], [2.0]), MvNormalMeanPrecision([3.0], [4.0]))
            q_p = (WishartFast(3.0, fill(3.0, 1, 1)), WishartFast(4.0, fill(5.0, 1, 1)))

            marginals = (
                Marginal(q_out, false, false, nothing),
                Marginal(q_switch, false, false, nothing),
                ManyOf(map(q_m_ -> Marginal(q_m_, false, false, nothing), q_m)),
                ManyOf(map(q_p_ -> Marginal(q_p_, false, false, nothing), q_p))
            )

            ref_val =
                0.5 * (score(AverageEnergy(), MvNormalMeanPrecision, Val{(:out, :μ, :Λ)}(), map((q) -> Marginal(q, false, false, nothing), (q_out, q_m[1], q_p[1])), nothing)) +
                0.5 * (score(AverageEnergy(), MvNormalMeanPrecision, Val{(:out, :μ, :Λ)}(), map((q) -> Marginal(q, false, false, nothing), (q_out, q_m[2], q_p[2])), nothing))
            @test score(AverageEnergy(), NormalMixture, Val{(:out, :switch, :m, :p)}(), marginals, nothing) ≈ ref_val
        end
    end
end

@testitem "nodes:NormalMixtureNode" begin
    using ReactiveMP, BayesBase, ExponentialFamily, Test

    import ReactiveMP:
        NormalMixtureNode,
        NormalMixtureNodeFactorisation,
        NormalMixtureNodeFunctionalDependencies,
        NormalMixture,
        factornode,
        functionalform,
        interfaceindex,
        interfaceindices,
        collect_functional_dependencies,
        collect_latest_marginals,
        collect_latest_messages,
        NodeInterface,
        IndexedNodeInterface,
        ManyOf,
        datavar,
        randomvar,
        sdtype,
        as_node_symbol,
        interfaces,
        alias_interface,
        collect_factorisation

    @testset "Construction and interface structure" begin
        interfaces = [(:out, datavar()), (:switch, randomvar()), (:m, randomvar()), (:m, randomvar()), (:p, randomvar()), (:p, randomvar())]
        factorizations = [[:out], [:switch], [:m1], [:m2], [:p1], [:p2]]

        node = factornode(NormalMixture, interfaces, factorizations)

        @test node isa NormalMixtureNode{2}
        @test sdtype(node) == Stochastic()
        @test functionalform(node) == NormalMixture{2}
        @test getinterfaces(node) isa Tuple
        @test length(getinterfaces(node)) == 6

        @test node.out isa NodeInterface
        @test node.switch isa NodeInterface
        @test all(i -> i isa IndexedNodeInterface, node.means)
        @test all(i -> i isa IndexedNodeInterface, node.precs)

        @test interfaceindex(node, :out) == 1
        @test interfaceindex(node, :switch) == 2
        @test interfaceindex(node, :m) == 3
        @test interfaceindex(node, :p) == 4
    end

    @testset "Construction errors" begin
        # not enough means/precisions
        @test_throws ErrorException factornode(NormalMixture, [(:out, :x), (:switch, :z), (:m, :m1), (:p, :p1)], [[:out], [:switch], [:m1], [:p1]])

        # mismatch counts
        @test_throws ErrorException factornode(NormalMixture, [(:out, :x), (:switch, :z), (:m, :m1), (:m, :m2), (:p, :p1)], [[:out], [:switch], [:m1], [:m2], [:p1]])

        # wrong factorization
        @test_throws ErrorException factornode(NormalMixture, [(:out, :x), (:switch, :z), (:m, :m1), (:m, :m2), (:p, :p1), (:p, :p2)], [[:out, :switch]])
    end

    @testset "Functional dependencies" begin
        interfaces = [(:out, datavar()), (:switch, randomvar()), (:m, randomvar()), (:m, randomvar()), (:p, randomvar()), (:p, randomvar())]
        factorizations = [[:out], [:switch], [:m1], [:m2], [:p1], [:p2]]
        node = factornode(NormalMixture, interfaces, factorizations)
        deps = NormalMixtureNodeFunctionalDependencies()

        # out dependencies
        msg_deps, marg_deps = functional_dependencies(deps, node, node.out, 1)
        @test msg_deps == ()
        @test length(marg_deps) == 3

        # switch dependencies
        msg_deps, marg_deps = functional_dependencies(deps, node, node.switch, 2)
        @test length(marg_deps) == 3

        # mean dependencies
        msg_deps, marg_deps = functional_dependencies(deps, node, node.means[1], 3)
        @test length(marg_deps) == 3

        # precision dependencies
        msg_deps, marg_deps = functional_dependencies(deps, node, node.precs[1], 5)
        @test length(marg_deps) == 3

        # invalid index
        @test_throws ErrorException functional_dependencies(deps, node, node.out, 99)
    end

    @testset "Collect functional dependencies" begin
        node = NormalMixtureNode(
            NodeInterface(:out, datavar()),
            NodeInterface(:switch, randomvar()),
            (IndexedNodeInterface(1, NodeInterface(:m, randomvar())), IndexedNodeInterface(2, NodeInterface(:m, randomvar()))),
            (IndexedNodeInterface(1, NodeInterface(:p, randomvar())), IndexedNodeInterface(2, NodeInterface(:p, randomvar())))
        )

        @test collect_functional_dependencies(node, nothing) isa NormalMixtureNodeFunctionalDependencies
        @test collect_functional_dependencies(node, NormalMixtureNodeFunctionalDependencies()) isa NormalMixtureNodeFunctionalDependencies
        @test_throws ErrorException collect_functional_dependencies(node, :wrong)
    end

    @testset "Collect latest marginals and messages" begin
        deps = NormalMixtureNodeFunctionalDependencies()
        node = NormalMixtureNode(
            NodeInterface(:out, datavar()),
            NodeInterface(:switch, randomvar()),
            (IndexedNodeInterface(1, NodeInterface(:m, randomvar())), IndexedNodeInterface(2, NodeInterface(:m, randomvar()))),
            (IndexedNodeInterface(1, NodeInterface(:p, randomvar())), IndexedNodeInterface(2, NodeInterface(:p, randomvar())))
        )

        # collect_latest_messages with empty tuple
        val, obs = collect_latest_messages(deps, node, ())
        @test val === nothing
        @test obs !== nothing

        # collect_latest_marginals with full (out, means, precs)
        marg_names, marg_obs = collect_latest_marginals(deps, node, (node.out, node.means, node.precs))
        @test marg_names isa Val
        @test !isnothing(marg_obs)

        # collect_latest_marginals with (out, switch, var)
        marg_names, marg_obs = collect_latest_marginals(deps, node, (node.out, node.switch, node.means[1]))
        @test marg_names isa Val
        @test !isnothing(marg_obs)
    end

    @testset "Type-level utilities" begin
        @test as_node_symbol(NormalMixture{2}) === :NormalMixture
        @test interfaces(NormalMixture{2}) === Val((:out, :switch, :m, :p))
        @test alias_interface(NormalMixture, 1, :m) === :m
        @test sdtype(NormalMixture{2}) == Stochastic()

        fact = collect_factorisation(NormalMixture{2}, :anything)
        @test fact isa NormalMixtureNodeFactorisation
    end

    @testset "Interface index utilities and unknown interface" begin
        node = NormalMixtureNode(
            NodeInterface(:out, datavar()),
            NodeInterface(:switch, randomvar()),
            (IndexedNodeInterface(1, NodeInterface(:m, randomvar())), IndexedNodeInterface(2, NodeInterface(:m, randomvar()))),
            (IndexedNodeInterface(1, NodeInterface(:p, randomvar())), IndexedNodeInterface(2, NodeInterface(:p, randomvar())))
        )

        @test interfaceindices(node, :out) == (1,)
        @test interfaceindices(node, :switch) == (2,)

        syms = (:out, :switch, :m, :p)
        idxs = interfaceindices(node, syms)
        @test idxs == map(s -> interfaceindex(node, s), syms)

        err = try
            interfaceindex(node, :nonexistent)
        catch e
            e
        end
        @test occursin("Unknown interface", sprint(showerror, err))
        @test occursin(string(functionalform(node)), sprint(showerror, err))
    end
end
