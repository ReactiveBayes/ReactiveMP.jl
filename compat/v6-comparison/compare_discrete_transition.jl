# DiscreteTransitionMessagePassingRules compared with v6's DiscreteTransition on identical
# inputs: every v6 method, the explicit ones and the generic ones, on inputs that select it, for
# belief propagation, mean-field, structured factorisations and joints of some of the `T`s, with
# up to four `T`s. The port has one rule per target, a tensor contraction over whatever inputs
# arrive; v6's explicit rules computed the same contraction, and agree up to the normalisation of
# the result, except the one declared below.
#
#   julia --startup-file=no --project=compat/v6-comparison compat/v6-comparison/compare_discrete_transition.jl

include(joinpath(@__DIR__, "V6Oracle.jl"))

using .V6Oracle, Test, Random
using ExponentialFamily, BayesBase, Distributions, LinearAlgebra
using MessagePassingRulesBase, MessagePassingRulesTestUtils, DiscreteTransitionMessagePassingRules
import ReactiveMP

const DT6 = ReactiveMP.DiscreteTransition

const SOFTMAX_DIMS = """
v6's rule towards `T3` of a five-interface node under belief propagation, with a
DirichletCollection q(a), exponentiated `E[log A]` with `softmax!(…; dims = 1)`, normalising each
column over `out` instead of the whole tensor, unlike every other rule of the node, so its message
weighted the columns differently. The port's single rule normalises globally, as all the others
did (user, 2026-09-24).
"""

# v6 names the interfaces `out`, `in`, `a`, `T1`, `T2`, …, a joint by its members joined with `_`,
# and takes the inputs in interface order, a joint at its first member.
v6_key(key::Symbol) = key
v6_key((group, k)::Tuple{Symbol, Int}) = Symbol(group, k)
v6_key(key::Tuple) = Symbol(join(map(v6_key, key), "_"))
interface_position(key::Symbol) = key === :out ? 1 : key === :in ? 2 : key === :a ? 3 : error("no interface `$key`")
interface_position((_, k)::Tuple{Symbol, Int}) = 3 + k
interface_position(key::Tuple) = interface_position(first(key))

function v6_named(inputs)
    sorted = sort(collect(inputs); by = pair -> interface_position(first(pair)))
    return NamedTuple{Tuple(map(v6_key ∘ first, sorted))}(Tuple(map(last, sorted)))
end

# The same inputs as a v7 rule takes them: a group as one tuple with `nothing` for the members
# left out, and joints as clusters.
function v7_named(inputs, n)
    singles = [key => value for (key, value) in inputs if key isa Symbol]
    members = Dict(last(key) => value for (key, value) in inputs if key isa Tuple{Symbol, Int})
    joints = Tuple(key => value for (key, value) in inputs if key isa Tuple && !(key isa Tuple{Symbol, Int}))
    named = NamedTuple{Tuple(map(first, singles))}(Tuple(map(last, singles)))
    isempty(members) || (named = merge(named, (T = ntuple(k -> get(members, k, nothing), n),)))
    return named, joints
end

# A v6 NamedTuple marginal as the port's FactorizedCluster: `in_T2` is `(:in, (:T, 2))`.
v6_member(name) = name == "out" ? :out : name == "in" ? :in : (:T, parse(Int, name[2:end]))
as_v7(v6::NamedTuple) = FactorizedCluster((Tuple(map(v6_member, split(String(key), "_"))) => value for (key, value) in pairs(v6))...)
as_v7(v6) = v6

function compare_message(id, target, m, q, n; declared = DeclaredDisagreement[])
    m7, _ = v7_named(m, n)
    q7, clusters = v7_named(q, n)
    v7 = getresult(call_message_update_rule(DiscreteTransition, target; m = m7, q = q7, clusters))
    v6, _ = v6_message_update(DT6, v6_key(target), v6_named(m), v6_named(q))
    return compare_with_reference(id, v7, v6; node = "DiscreteTransition", target = repr(target), declared).outcome
end

function compare_marginal(id, members, m, q, n)
    m7, _ = v7_named(m, n)
    q7, clusters = v7_named(q, n)
    v7 = getresult(call_marginal_update_rule(DiscreteTransition, members; m = m7, q = q7, clusters))
    v6 = v6_marginal_update(DT6, map(v6_key, members), v6_named(m), v6_named(q))
    return compare_with_reference(id, v7, as_v7(v6); node = "DiscreteTransition", target = repr(members)).outcome
end

# `generic = true` calls v6's generic energy past its explicit methods, which failed where the
# port does not: the one of a joint over three or more axes with a DirichletCollection q(a) was
# ambiguous with the generic one, and the one of `q(out, in)` with a point-mass q(a) asserted a
# square tensor.
function compare_energy(id, q, n; generic = false)
    q7, clusters = v7_named(q, n)
    v7 = getresult(call_average_energy(DiscreteTransition; q = q7, clusters))
    inputs = v6_named(q)
    names = Val(keys(inputs))
    marginals = map(value -> ReactiveMP.Marginal(value, false, false), Tuple(values(inputs)))
    v6 = if generic
        types = Tuple{map(value -> ReactiveMP.Marginal{<:(value isa PointMass ? typeof(value) : typeof(value).name.wrapper)}, Tuple(values(inputs)))...}
        invoke(ReactiveMP.score, Tuple{ReactiveMP.AverageEnergy, Type{DT6}, typeof(names), types, Nothing}, ReactiveMP.AverageEnergy(), DT6, names, marginals, nothing)
    else
        ReactiveMP.score(ReactiveMP.AverageEnergy(), DT6, names, marginals, nothing)
    end
    return compare_with_reference(id, v7, v6; node = "DiscreteTransition", target = "energy").outcome
end

# v6 read a Bernoulli `in` or `out` with `probvec`, a tuple its tensor algebra did not take; the
# port takes it, and is compared with v6 on the same distribution as a Categorical.
as_categorical(inputs) = [key => (value isa Bernoulli ? Categorical(collect(probvec(value))) : value) for (key, value) in inputs]

rng = Xoshiro(42)
categorical(k) = Categorical(normalize!(rand(rng, k) .+ 0.1, 1))
onehot(k, i) = PointMass([j == i ? 1.0 : 0.0 for j in 1:k])
contingency(dims...) = Contingency(normalize!(rand(rng, dims...) .+ 0.1, 1))
# The sizes of `out`, `in` and each `T`, and q(a) as a point mass of a conditional tensor or as
# a DirichletCollection.
const SIZES = (3, 2, 4, 2, 3, 2)
sizes(n) = SIZES[1:(2 + n)]
function tensor_marginals(n)
    A = rand(rng, sizes(n)...) .+ 0.05
    A ./= sum(A; dims = 1)
    return ("point mass" => PointMass(A), "Dirichlet" => DirichletCollection(rand(rng, sizes(n)...) .+ 0.5))
end
interface_keys(n) = (:out, :in, ntuple(k -> (:T, k), n)...)
interface_size(key, n) = sizes(n)[key === :out ? 1 : key === :in ? 2 : 2 + last(key)]

@testset "DiscreteTransitionMessagePassingRules against v6" begin
    # Belief propagation: every other categorical interface sends a message. v6 had explicit
    # rules for up to three `T`s, and its generic one for more.
    @testset "belief propagation" begin
        for n in 0:4, (label, q_a) in tensor_marginals(n), target in interface_keys(n)
            m = [key => categorical(interface_size(key, n)) for key in interface_keys(n) if key != target]
            id = "BP:$n T:$label:$(repr(target))"
            declared = (n == 3 && target == (:T, 3) && q_a isa DirichletCollection) ? [DeclaredDisagreement(id; kind = :correction, reasoning = SOFTMAX_DIMS)] : DeclaredDisagreement[]
            @test compare_message(id, target, m, [:a => q_a], n; declared) === (isempty(declared) ? :agree : :correction)
        end
    end

    # Mean-field: every input a marginal, categorical or a point mass; a Bernoulli `in`.
    @testset "mean-field" begin
        for n in 0:3, (label, q_a) in tensor_marginals(n), target in interface_keys(n)
            q = [key => categorical(interface_size(key, n)) for key in interface_keys(n) if key != target]
            @test compare_message("MF:$n T:$label:$(repr(target))", target, [], [q..., :a => q_a], n) === :agree
            observed = [key => onehot(interface_size(key, n), 1) for key in interface_keys(n) if key != target]
            @test compare_message("MF observed:$n T:$label:$(repr(target))", target, [], [observed..., :a => q_a], n) === :agree
        end
        for (label, q_a) in tensor_marginals(1)
            q = [:in => Bernoulli(0.3), (:T, 1) => categorical(4), :a => q_a]
            q7, _ = v7_named(q, 1)
            v7 = getresult(call_message_update_rule(DiscreteTransition, :out; q = q7))
            v6, _ = v6_message_update(DT6, :out, NamedTuple(), v6_named(as_categorical(q)))
            @test compare_with_reference("MF Bernoulli in:$label", v7, v6; node = "DiscreteTransition", target = ":out").outcome === :agree
        end
    end

    # Messages beside marginals: an observed `out` or `T1` as a marginal, the others sending
    # messages (v6's explicit rules for these), and every other mix.
    @testset "messages and marginals" begin
        for n in 0:3, (label, q_a) in tensor_marginals(n), target in interface_keys(n)
            target === :out && continue
            m = [key => categorical(interface_size(key, n)) for key in interface_keys(n) if key != target && key != :out]
            @test compare_message("observed out:$n T:$label:$(repr(target))", target, m, [:out => onehot(sizes(n)[1], 2), :a => q_a], n) === :agree
        end
        for (label, q_a) in tensor_marginals(1), target in (:out, :in)
            m = [key => categorical(interface_size(key, 1)) for key in (:out, :in) if key != target]
            @test compare_message("observed T1:$label:$(repr(target))", target, m, [(:T, 1) => onehot(4, 3), :a => q_a], 1) === :agree
        end
        for (label, q_a) in tensor_marginals(2), target in interface_keys(2)
            others = [key for key in interface_keys(2) if key != target]
            m = [key => categorical(interface_size(key, 2)) for key in others[1:1]]
            q = [key => categorical(interface_size(key, 2)) for key in others[2:end]]
            @test compare_message("mixed:$label:$(repr(target))", target, m, [q..., :a => q_a], 2) === :agree
        end
    end

    # Structured factorisations: joints over the target's neighbours, and joints of some of the
    # `T`s with `out` or `in`.
    @testset "structured" begin
        for (label, q_a) in tensor_marginals(1)
            @test compare_message("q(out, in):$label:T1", (:T, 1), [], [(:out, :in) => contingency(3, 2), :a => q_a], 1) === :agree
            @test compare_message("q(out, T1):$label:in", :in, [], [(:out, (:T, 1)) => contingency(3, 4), :a => q_a], 1) === :agree
            @test compare_message("q(in, T1), m(out):$label:in", :in, [:out => categorical(3)], [(:T, 1) => categorical(4), :a => q_a], 1) === :agree
        end
        for (label, q_a) in tensor_marginals(2)
            @test compare_message("q(T1, T2), m(out):$label:in", :in, [:out => categorical(3)], [((:T, 1), (:T, 2)) => contingency(4, 2), :a => q_a], 2) === :agree
            @test compare_message("q(out, T1), q(T2):$label:in", :in, [], [(:out, (:T, 1)) => contingency(3, 4), (:T, 2) => categorical(2), :a => q_a], 2) === :agree
            @test compare_message("q(out, in, T2), q(T1):$label:T2", (:T, 2), [:out => categorical(3), :in => categorical(2)], [(:T, 1) => categorical(4), :a => q_a], 2) === :agree
            @test compare_message("q(out, in), q(T1, T2):$label:T1", (:T, 1), [(:T, 2) => categorical(2)], [(:out, :in) => contingency(3, 2), :a => q_a], 2) === :agree
            @test compare_message("q(in, T2), m(out):$label:T1", (:T, 1), [:out => categorical(3)], [(:in, (:T, 2)) => contingency(2, 2), :a => q_a], 2) === :agree
        end
        for (label, q_a) in tensor_marginals(3)
            @test compare_message("q(in, T1, T3):$label:T2", (:T, 2), [:out => categorical(3)], [(:in, (:T, 1), (:T, 3)) => contingency(2, 4, 3), :a => q_a], 3) === :agree
        end
    end

    # Towards `a`: the expected counts, from the marginals of every other interface.
    @testset "towards a" begin
        @test compare_message("a:q(out) observed, q(in)", :a, [], [:out => onehot(3, 1), :in => categorical(2)], 0) === :agree
        @test compare_message("a:q(out, in)", :a, [], [(:out, :in) => contingency(3, 2)], 0) === :agree
        @test compare_message("a:q(out, in, T1)", :a, [], [(:out, :in, (:T, 1)) => contingency(3, 2, 4)], 1) === :agree
        @test compare_message("a:q(out, in, T1, T2)", :a, [], [(:out, :in, (:T, 1), (:T, 2)) => contingency(3, 2, 4, 2)], 2) === :agree
        @test compare_message("a:q(out, in), q(T1)", :a, [], [(:out, :in) => contingency(3, 2), (:T, 1) => categorical(4)], 1) === :agree
        @test compare_message("a:q(out, in), q(T1) observed", :a, [], [(:out, :in) => contingency(3, 2), (:T, 1) => onehot(4, 2)], 1) === :agree
        @test compare_message("a:q(out, T1), q(in)", :a, [], [(:out, (:T, 1)) => contingency(3, 4), :in => categorical(2)], 1) === :agree
        @test compare_message("a:q(in, T2), q(out), q(T1)", :a, [], [(:in, (:T, 2)) => contingency(2, 2), :out => categorical(3), (:T, 1) => categorical(4)], 2) === :agree
        for n in 0:3
            @test compare_message("a:MF:$n T", :a, [], [key => categorical(interface_size(key, n)) for key in interface_keys(n)], n) === :agree
        end
    end

    # The marginals of clusters: the messages of the members, and the marginals of the rest.
    # Observed members are split off as blocks, as v6 did.
    @testset "marginals" begin
        for (label, q_a) in tensor_marginals(0)
            @test compare_marginal("(out, in):$label", (:out, :in), [:out => categorical(3), :in => categorical(2)], [:a => q_a], 0) === :agree
            @test compare_marginal("(out, in), out observed:$label", (:out, :in), [:out => onehot(3, 2), :in => categorical(2)], [:a => q_a], 0) === :agree
            @test compare_marginal("(out, in), all observed:$label", (:out, :in), [:out => onehot(3, 2), :in => onehot(2, 1)], [:a => q_a], 0) === :agree
        end
        for (label, q_a) in tensor_marginals(1)
            m = [:out => categorical(3), :in => categorical(2)]
            @test compare_marginal("(out, in), q(T1) observed:$label", (:out, :in), m, [:a => q_a, (:T, 1) => onehot(4, 3)], 1) === :agree
            @test compare_marginal("(out, in), q(T1):$label", (:out, :in), m, [:a => q_a, (:T, 1) => categorical(4)], 1) === :agree
            @test compare_marginal("(out, in, T1):$label", (:out, :in, (:T, 1)), [m..., (:T, 1) => categorical(4)], [:a => q_a], 1) === :agree
            @test compare_marginal("(out, in, T1), out observed:$label", (:out, :in, (:T, 1)), [:out => onehot(3, 1), :in => categorical(2), (:T, 1) => categorical(4)], [:a => q_a], 1) === :agree
            @test compare_marginal("(out, in, T1), in observed:$label", (:out, :in, (:T, 1)), [:out => categorical(3), :in => onehot(2, 2), (:T, 1) => categorical(4)], [:a => q_a], 1) === :agree
            @test compare_marginal("(in, T1), q(out):$label", (:in, (:T, 1)), [:in => categorical(2), (:T, 1) => categorical(4)], [:out => categorical(3), :a => q_a], 1) === :agree
        end
        for (label, q_a) in tensor_marginals(2)
            all_members = (:out, :in, (:T, 1), (:T, 2))
            @test compare_marginal("(out, in, T1, T2):$label", all_members, [key => categorical(interface_size(key, 2)) for key in all_members], [:a => q_a], 2) === :agree
            observed = [:out => onehot(3, 3), :in => categorical(2), (:T, 1) => onehot(4, 2), (:T, 2) => categorical(2)]
            @test compare_marginal("(out, in, T1, T2), out and T1 observed:$label", all_members, observed, [:a => q_a], 2) === :agree
            @test compare_marginal("(out, T1, T2), q(in):$label", (:out, (:T, 1), (:T, 2)), [:out => categorical(3), (:T, 1) => categorical(4), (:T, 2) => categorical(2)], [:in => categorical(2), :a => q_a], 2) === :agree
            @test compare_marginal("(T1, T2), q(out, in):$label", ((:T, 1), (:T, 2)), [(:T, 1) => categorical(4), (:T, 2) => categorical(2)], [(:out, :in) => contingency(3, 2), :a => q_a], 2) === :agree
            @test compare_marginal("(T1, T2), T1 observed, q(out, in):$label", ((:T, 1), (:T, 2)), [(:T, 1) => onehot(4, 4), (:T, 2) => categorical(2)], [(:out, :in) => contingency(3, 2), :a => q_a], 2) === :agree
            @test compare_marginal("(out, T2), q(in), q(T1):$label", (:out, (:T, 2)), [:out => categorical(3), (:T, 2) => categorical(2)], [:in => categorical(2), (:T, 1) => categorical(4), :a => q_a], 2) === :agree
        end

        # The engine keys a joint over the whole group `T` by the group, and an observed member of
        # it stays in the joint as a one-hot axis: the same distribution as v6's blocks.
        for (label, q_a) in tensor_marginals(1)
            m = [:out => categorical(3), :in => categorical(2), (:T, 1) => onehot(4, 2)]
            m7, _ = v7_named(m, 1)
            v7 = getresult(call_marginal_update_rule(DiscreteTransition, (:out, :in, :T); m = m7, q = (a = q_a,)))
            v6 = v6_marginal_update(DT6, (:out, :in, :T1), v6_named(m), v6_named([:a => q_a]))
            joint = components(v6.out_in) .* reshape(mean(v6.T1), 1, 1, 4)
            @test v7 isa Contingency && components(v7) ≈ joint
        end
    end

    @testset "average energy" begin
        for (label, q_a) in tensor_marginals(0)
            @test compare_energy("q(out) observed, q(in):$label", [:out => onehot(3, 2), :in => categorical(2), :a => q_a], 0) === :agree
            @test compare_energy("q(out), q(in):$label", [:out => categorical(3), :in => categorical(2), :a => q_a], 0) === :agree
            @test compare_energy("q(out, in):$label", [(:out, :in) => contingency(3, 2), :a => q_a], 0; generic = q_a isa PointMass) === :agree
            A = rand(rng, 2, 2)
            @test compare_energy("q(out, in), square:$label", [(:out, :in) => contingency(2, 2), :a => q_a isa PointMass ? PointMass(A ./ sum(A; dims = 1)) : DirichletCollection(A .+ 1)], 0) === :agree
        end
        for n in 1:3, (label, q_a) in tensor_marginals(n)
            @test compare_energy("MF:$n T:$label", [[key => categorical(interface_size(key, n)) for key in interface_keys(n)]..., :a => q_a], n) === :agree
            joint = (:out, :in, ntuple(k -> (:T, k), n)...)
            @test compare_energy("joint:$n T:$label", [joint => contingency(sizes(n)...), :a => q_a], n; generic = true) === :agree
            @test compare_energy("q(out, in), q(T…):$n T:$label", [(:out, :in) => contingency(3, 2), [(:T, k) => categorical(sizes(n)[2 + k]) for k in 1:n]..., :a => q_a], n) === :agree
        end
        for (label, q_a) in tensor_marginals(2)
            @test compare_energy("q(out, T2), q(in), q(T1):$label", [(:out, (:T, 2)) => contingency(3, 2), :in => categorical(2), (:T, 1) => onehot(4, 1), :a => q_a], 2) === :agree
            q = [:out => categorical(3), :in => Bernoulli(0.7), (:T, 1) => categorical(4), (:T, 2) => categorical(2), :a => q_a]
            q7, _ = v7_named(q, 2)
            v6 = compare_energy("Bernoulli in, as v6's Categorical:$label", as_categorical(q), 2)
            @test v6 === :agree
            @test getresult(call_average_energy(DiscreteTransition; q = q7)) ≈ getresult(call_average_energy(DiscreteTransition; q = first(v7_named(as_categorical(q), 2))))
        end
    end
end
