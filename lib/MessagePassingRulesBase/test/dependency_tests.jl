@testmodule DependencyNodes begin
    using MessagePassingRulesBase
    using MessagePassingRulesBase: AbstractAlgorithm, DefaultAlgorithmExtension, select_group_members

    # The mixture's own algorithm: its rules ignore the factorisation.
    struct MixtureVMP <: AbstractAlgorithm end

    struct NormalMixture end
    @define_factor_node(
        node = NormalMixture,
        type = Stochastic,
        interfaces = [:out, :switch, :m..., :p...],
        algorithm = MixtureVMP,
        dependencies = [
            (:m, k) => (q[:out], q[:switch], q[:p][k]),
            (:p, k) => (q[:out], q[:switch], q[:m][k]),
        ],
    )

    struct DeltaFn end
    @define_factor_node(node = DeltaFn, type = Deterministic, interfaces = [:out, :in...], static_inputs = :fold)

    struct ToyDelta <: AbstractAlgorithm end
    @define_dependencies(
        node = DeltaFn,
        algorithm = ToyDelta,
        dependencies = [
            :out => (m[:in...],),
            (:in, k) => (m[:out], m[:in][!k]),
        ],
    )

    struct Chain <: AbstractAlgorithm end
    @define_dependencies(
        node = DeltaFn,
        algorithm = Chain,
        dependencies = [(:in, k) => (m[:in][select_group_members(j -> (mod1(j - 1, 3),); arity = 1)],)],
    )

    # A standalone algorithm that fixes what the node's rules consume and what free energy is
    # computed over, whatever the factorisation.
    struct FixedPartition <: AbstractAlgorithm end
    struct Gauss end
    @define_factor_node(node = Gauss, type = Stochastic, interfaces = [:out, :μ, :τ])
    @define_dependencies(
        node = Gauss,
        algorithm = FixedPartition,
        dependencies = [:τ => (q[:out, :μ],), :out => (m[:μ], q[:τ])],
        free_energy_partition = [(:out, :μ), (:τ,)],
    )

    # An algorithm that extends the default scheme: the rule towards `a` also reads `q(a)`, and
    # the other inputs follow the factorisation, mean-field or `q(y, x)`.
    struct Transition end
    struct TransitionVMP <: AbstractAlgorithm end
    @define_factor_node(node = Transition, type = Stochastic, interfaces = [:y, :x, :a, :W])
    @define_dependencies(
        node = Transition, algorithm = TransitionVMP,
        dependencies = [:y => (default,), :x => default, :a => (default, q[:a]), :W => (default,)],
    )
    @define_message_update_rule(node = Transition, target = :a, algorithm = TransitionVMP, args = (q[:y]::Any, q[:x]::Any, q[:a]::Any, q[:W]::Any), body = (args) -> 1)
    @define_message_update_rule(node = Transition, target = :a, algorithm = TransitionVMP, args = (q[:y, :x]::Any, q[:a]::Any, q[:W]::Any), body = (args) -> 2)
    @define_message_update_rule(node = Transition, target = :y, algorithm = TransitionVMP, args = (q[:x]::Any, q[:a]::Any, q[:W]::Any), body = (args) -> 3)
end

@testitem "dependencies:declared" tags = [:base] setup = [DependencyNodes] begin
    using MessagePassingRulesBase: dependencies_spec, Target, IndexedTarget, DefaultAlgorithm, target_dependencies,
        Dependency, AllGroupMembers, AlignedGroupMember, AllGroupMembersButSelf, SingleInterface, CustomGroupSelector, static_inputs, free_energy_partition
    D = DependencyNodes

    # The node's own declaration applies to its default algorithm.
    declaration = dependencies_spec(D.NormalMixture, D.MixtureVMP())
    @test declaration !== nothing
    m_k = target_dependencies(declaration, IndexedTarget(:m, 2))
    @test map(d -> (d.container, d.key), m_k) == ((:q, :out), (:q, :switch), (:q, :p))
    @test m_k[3].selector isa AlignedGroupMember

    # Another algorithm, another declaration; an undeclared algorithm has none.
    delta = dependencies_spec(D.DeltaFn, D.ToyDelta())
    @test only(target_dependencies(delta, Target(:out))).selector isa AllGroupMembers
    @test target_dependencies(delta, IndexedTarget(:in, 1))[2].selector isa AllGroupMembersButSelf
    @test dependencies_spec(D.DeltaFn, DefaultAlgorithm()) === nothing
    @test target_dependencies(delta, Target(:nope)) === nothing

    @test static_inputs(D.DeltaFn) === :fold
    @test static_inputs(D.NormalMixture) === :none

    # Consumed and scored are separate: `τ` consumes the joint, the free_energy_partition says what is scored.
    fixed = dependencies_spec(D.Gauss, D.FixedPartition())
    τ = only(target_dependencies(fixed, Target(:τ)))
    @test τ.key === (:out, :μ)
    @test free_energy_partition(fixed) === ((:out, :μ), (:τ,))
    @test free_energy_partition(declaration) === nothing
end

@testitem "dependencies:extending the default scheme" tags = [:base] setup = [DependencyNodes] begin
    using MessagePassingRulesBase: dependencies_spec, Target, target_dependencies, extends_default_scheme, SingleInterface
    D = DependencyNodes

    # `default` is the default scheme's inputs; the target's listed inputs are added to them.
    declaration = dependencies_spec(D.Transition, D.TransitionVMP())
    a = only(target_dependencies(declaration, Target(:a)))
    @test (a.container, a.key) === (:q, :a) && a.selector isa SingleInterface
    @test target_dependencies(declaration, Target(:y)) === ()
    @test target_dependencies(declaration, Target(:x)) === ()
    @test all(t -> extends_default_scheme(declaration, Target(t)), (:y, :x, :a, :W))

    # A declaration without `default` replaces the default scheme; an undeclared target has none.
    fixed = dependencies_spec(D.Gauss, D.FixedPartition())
    @test !extends_default_scheme(fixed, Target(:τ))
    @test !extends_default_scheme(fixed, Target(:μ))

    # The display names the default scheme.
    plain = sprint(show, MIME("text/plain"), declaration)
    @test contains(plain, ":a ⇐ default, q[:a]")
    @test contains(plain, ":y ⇐ default")
    # The card draws the default scheme's inputs and the added marginal as two edges.
    html = sprint(show, MIME("text/html"), declaration)
    @test contains(html, "class=\"edge default\"") && contains(html, ">q[:a]</text>")
end

@testitem "dependencies:selectors" tags = [:base] setup = [DependencyNodes] begin
    using MessagePassingRulesBase: select_group_members, selected_indices, AllGroupMembers, AlignedGroupMember, AllGroupMembersButSelf, selection_arity
    @test selected_indices(AllGroupMembers(), 2, 4) === (1, 2, 3, 4)
    @test selected_indices(AlignedGroupMember(), 2, 4) === (2,)
    @test selected_indices(AllGroupMembersButSelf(), 2, 4) === (1, 3, 4)
    @test selection_arity(AllGroupMembersButSelf(), 4) == 3
    @test selection_arity(AlignedGroupMember(), 4) == 1

    # A one-member group selects nothing, as a value: an empty tuple, never a stall.
    @test selected_indices(AllGroupMembersButSelf(), 1, 1) === ()
    @test selection_arity(AllGroupMembersButSelf(), 1) == 0

    chain = select_group_members(j -> (mod1(j - 1, 3),); arity = 1)
    @test selected_indices(chain, 1, 3) === (3,)
    @test selection_arity(chain, 3) == 1
    # Static arity is enforced: a selector that returns a different length is an error.
    bad = select_group_members(j -> j == 1 ? () : (j - 1,); arity = 1)
    @test_throws ArgumentError selected_indices(bad, 1, 3)
end

@testitem "dependencies:malformed" tags = [:base] begin
    using MessagePassingRulesBase

    function failure(ex)
        err = try
            Core.eval(Module(), Expr(:toplevel, :(using MessagePassingRulesBase), :(using MessagePassingRulesBase: AbstractAlgorithm), ex))
            nothing
        catch e
            e isa LoadError ? e.error : e
        end
        return err === nothing ? "" : sprint(showerror, err)
    end
    node(deps; kw...) = :(struct N end; @define_factor_node(node = N, type = Stochastic, interfaces = [:out, :μ, :p...], dependencies = $deps))

    @test failure(node(:([:nope => (m[:μ],)]))) |> msg -> contains(msg, "no interface `nope`")
    @test failure(node(:([:out => (m[:nope],)]))) |> msg -> contains(msg, "no interface `nope`")
    @test failure(node(:([(:μ, k) => (m[:out],)]))) |> msg -> contains(msg, "`μ` is not a group")
    @test failure(node(:([:p => (m[:out],)]))) |> msg -> contains(msg, "`p` is a group")
    @test failure(node(:([:out => (m[:p],)]))) |> msg -> contains(msg, "group `p`")
    @test failure(node(:([:out => (m[:p][k],)]))) |> msg -> contains(msg, "needs an indexed target")
    @test failure(node(:([(:p, k) => (m[:μ][k],)]))) |> msg -> contains(msg, "`μ` is not a group")
    @test failure(node(:([(:p, k) => (m[:p][j],)]))) |> msg -> contains(msg, "`j` is not the index")
    @test failure(node(:([:out => (q[:μ, :out],)]))) |> msg -> contains(msg, "interface order")
    # A cluster may hold members of a group by index, as a rule's input does.
    @test failure(node(:([:out => (q[:μ, (:out, 1)],)]))) |> msg -> contains(msg, "`out` is not a group")
    @test failure(node(:([:out => (q[:μ, (:p, 0)],)]))) |> msg -> contains(msg, "indexed from 1")
    @test failure(node(:([:out => (q[:μ, (:p, 2), (:p, 1)],)]))) |> msg -> contains(msg, "a group's by index")
    @test failure(node(:([:out => (q[:μ, (:p, k)],)]))) |> msg -> contains(msg, "with a literal index")
    @test failure(node(:([:out => (m[:μ], m[:μ])]))) |> msg -> contains(msg, "twice")
    @test failure(node(:([:out => (m[:μ],), :out => (q[:μ],)]))) |> msg -> contains(msg, "declared twice")
    # `default` once, and beside it only inputs of a single interface.
    @test failure(node(:([:out => (default, m[:μ], default)]))) |> msg -> contains(msg, "`default` twice")
    @test failure(node(:([:out => (default, q[:out, :μ])]))) |> msg -> contains(msg, "extends the default scheme with a single interface")
    @test failure(node(:([:out => (default, q[:p...])]))) |> msg -> contains(msg, "extends the default scheme with a single interface")
    @test failure(:(struct N end; @define_factor_node(node = N, type = Stochastic, interfaces = [:out], static_inputs = :sometimes))) |>
        msg -> contains(msg, "`static_inputs` must be :none or :fold")
    @test failure(
        :(
            struct N end; struct A <: AbstractAlgorithm end;
            @define_factor_node(node = N, type = Stochastic, interfaces = [:out, :μ]);
            @define_dependencies(node = N, algorithm = A, dependencies = [:out => (m[:μ],)], free_energy_partition = [(:out,), (:out, :μ)])
        )
    ) |> msg -> contains(msg, "`out` appears in more than one")
    @test failure(
        :(
            struct N end; struct A <: AbstractAlgorithm end;
            @define_factor_node(node = N, type = Stochastic, interfaces = [:out, :μ]);
            @define_dependencies(node = N, algorithm = A, dependencies = [:out => (m[:μ],)], free_energy_partition = [(:out,)])
        )
    ) |> msg -> contains(msg, "does not cover `μ`")
end

@testitem "dependencies:a custom selection shows as declared" tags = [:base] setup = [DependencyNodes] begin
    using MessagePassingRulesBase: dependencies_spec, target_dependencies, IndexedTarget
    spec = dependencies_spec(DependencyNodes.DeltaFn, DependencyNodes.Chain())
    @test sprint(show, only(target_dependencies(spec, IndexedTarget(:in, 1)))) == "m[:in][select_group_members(…; arity = 1)]"
    @test contains(sprint(show, MIME"text/html"(), spec), "select_group_members(…; arity = 1)")
end

@testitem "dependencies:a joint holding a group member" tags = [:base] begin
    using MessagePassingRulesBase
    using MessagePassingRulesBase: dependencies_spec, target_dependencies, Target, IndexedTarget, SingleInterface

    struct Tensor end
    struct MemberJoint <: AbstractAlgorithm end
    @define_factor_node(node = Tensor, type = Stochastic, interfaces = [:out, :in, :T...])
    # The joint of `out` and the first member of `T`, as a rule takes it, and one of two members.
    @define_dependencies(
        node = Tensor, algorithm = MemberJoint,
        dependencies = [:out => (default,), :in => (q[:out, (:T, 1), (:T, 2)],), (:T, k) => (q[:out, (:T, 1)], m[:in])],
    )
    spec = dependencies_spec(Tensor, MemberJoint())
    joint, message = target_dependencies(spec, IndexedTarget(:T, 2))
    @test joint.container === :q && joint.key == (:out, (:T, 1)) && joint.selector isa SingleInterface
    @test message.key === :in
    @test only(target_dependencies(spec, Target(:in))).key == (:out, (:T, 1), (:T, 2))
    # It shows as it is written, in the terminal and on the card.
    @test sprint(show, joint) == "q[:out, (:T, 1)]"
    @test contains(sprint(show, MIME"text/plain"(), spec), "⇐ q[:out, (:T, 1)], m[:in]")
    @test contains(sprint(show, MIME"text/html"(), spec), ">q[:out, (:T, 1)]</text>")
end
