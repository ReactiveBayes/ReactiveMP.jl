# A marginal rule whose cluster factorises returns a `FactorizedCluster`: its blocks are keyed
# by tuples of members, carried in the type, never by a mangled name like `out_μ`.

@testitem "factorized-cluster:access" tags = [:base] begin
    using MessagePassingRulesBase: FactorizedCluster, cluster_blocks
    import BayesBase

    fc = FactorizedCluster((:out, :μ) => "joint", (:v,) => "v")
    @test cluster_blocks(fc) === ((:out, :μ), (:v,))
    @test fc[(:out, :μ)] === "joint"
    @test fc[(:v,)] === "v"
    @test_throws KeyError fc[(:μ,)]
    @test collect(pairs(fc)) == [(:out, :μ) => "joint", (:v,) => "v"]
    @test_throws ArgumentError FactorizedCluster()
    # The blocks are a BayesBase `FactorizedJoint`: the labels are all that is added.
    @test fc.joint isa BayesBase.FactorizedJoint
    @test BayesBase.components(fc) === ("joint", "v")
    # A block may hold members of a group, as a joint over some of them is keyed.
    fc = FactorizedCluster((:out,) => "out", (:in, (:T, 2)) => "joint", ((:T, 1),) => "T1")
    @test fc[(:in, (:T, 2))] === "joint" && fc[((:T, 1),)] === "T1"
end

@testmodule FactorizedBodies begin
    using MessagePassingRulesBase: FactorizedCluster

    # `const`, as a rule body is: a non-const binding measures dynamic dispatch instead.
    const build = (a, b) -> FactorizedCluster((:out, :μ) => a, (:v,) => b)
    const read = (fc) -> fc[(:out, :μ)] + fc[(:v,)]
    # Fixed arity: a varargs helper that splats adds 48 bytes on 1.10.
    measure_build(a, b) = (build(a, b); @allocated build(a, b))
    measure_read(fc) = (read(fc); @allocated read(fc))
end

@testitem "gate:factorized-cluster" tags = [:base, :alloc] setup = [FactorizedBodies] begin
    using MessagePassingRulesBase: FactorizedCluster
    F = FactorizedBodies

    # Built in a rule body from literal keys, the blocks are known to the compiler.
    @test (@inferred F.build(1.0, 2.0)) isa FactorizedCluster{((:out, :μ), (:v,))}
    fc = F.build(1.0, 2.0)
    @test (@inferred F.read(fc)) === 3.0
    @test F.measure_build(1.0, 2.0) == 0
    @test F.measure_read(fc) == 0
end

@testitem "factorized-cluster:partition" tags = [:base] begin
    using MessagePassingRulesBase: FactorizedCluster, ClusterTarget, check_factorized_cluster

    target = ClusterTarget((:out, :μ, :v))
    fc = FactorizedCluster((:out, :μ) => 1, (:v,) => 2)
    @test check_factorized_cluster(target, fc) === fc
    @test check_factorized_cluster(target, FactorizedCluster((:out,) => 1, (:μ,) => 2, (:v,) => 3)) isa FactorizedCluster

    message(f) = try
        f(); ""
    catch e
        sprint(showerror, e)
    end
    @test contains(message(() -> check_factorized_cluster(target, FactorizedCluster((:out, :μ) => 1))), "does not cover `v`")
    @test contains(message(() -> check_factorized_cluster(target, FactorizedCluster((:out, :μ) => 1, (:μ, :v) => 2))), "`μ` is in more than one block")
    @test contains(message(() -> check_factorized_cluster(target, FactorizedCluster((:μ, :out) => 1, (:v,) => 2))), "cluster order")
    @test contains(message(() -> check_factorized_cluster(target, FactorizedCluster((:out, :μ, :v, :τ) => 1))), "`τ` is not a member")
end

@testitem "factorized-cluster:entropy" tags = [:base] begin
    using MessagePassingRulesBase: FactorizedCluster
    import BayesBase

    struct WithEntropy
        h::Float64
    end
    BayesBase.entropy(d::WithEntropy) = d.h

    # Independent blocks: the entropy of the whole is the sum over the blocks.
    @test BayesBase.entropy(FactorizedCluster((:out, :μ) => WithEntropy(1.5), (:v,) => WithEntropy(0.25))) == 1.75
end

@testitem "factorized-cluster:rule" tags = [:base] begin
    using MessagePassingRulesBase
    using MessagePassingRulesBase: RuleArgs, ClusterTarget, DefaultAlgorithm, FactorizedCluster, check_factorized_cluster
    using BayesBase: PointMass

    struct Gauss end
    @define_factor_node(node = Gauss, type = Stochastic, interfaces = [:out, :μ, :v], algorithm = DefaultAlgorithm)
    # q(out, μ, v) = q(out, μ) q(v).
    @define_marginal_update_rule(
        node = Gauss, target = (:out, :μ, :v),
        args = (m[:out]::Float64, m[:μ]::Float64, m[:v]::Float64),
        body = (args) -> FactorizedCluster((:out, :μ) => PointMass([args.m[:out], args.m[:μ]]), (:v,) => PointMass(args.m[:v])),
    )
    target = ClusterTarget((:out, :μ, :v))
    result = message_passing_marginalrule(Gauss, target, DefaultAlgorithm(), RuleArgs(m = (out = 1.0, μ = 2.0, v = 3.0)))
    @test check_factorized_cluster(target, result)[(:out, :μ)].point == [1.0, 2.0]
end

@testitem "factorized-cluster:float-type" tags = [:base] begin
    using MessagePassingRulesBase: FactorizedCluster
    using BayesBase: PointMass, paramfloattype, convert_paramfloattype, mean

    # BayesBase's parameter float-type interface, as for its own `FactorizedJoint`: what the
    # type-promotion checks of the test tables rely on.
    fc = FactorizedCluster((:out, :μ) => PointMass([1.0f0, 2.0f0]), (:v,) => PointMass(3.0))
    @test paramfloattype(fc) === Float64
    converted = convert_paramfloattype(BigFloat, fc)
    @test converted isa FactorizedCluster{((:out, :μ), (:v,))}
    @test converted[(:out, :μ)] isa PointMass{Vector{BigFloat}}
    @test converted[(:v,)] isa PointMass{BigFloat}
    @test mean(converted[(:v,)]) == 3
end
