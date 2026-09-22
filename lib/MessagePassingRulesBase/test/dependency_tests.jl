@testmodule DependencyNodes begin
    using MessagePassingRulesBase
    using MessagePassingRulesBase: VMP, BP, AbstractAlgorithm, select_group_members

    struct NormalMixture end
    @define_factor_node(
        node = NormalMixture,
        type = Stochastic,
        interfaces = [:out, :switch, :m..., :p...],
        algorithm = VMP,
        dependencies = [
            (:m, k) => (q[:out], q[:switch], q[:p][k]),
            (:p, k) => (q[:out], q[:switch], q[:m][k]),
        ],
    )

    struct DeltaFn end
    @define_factor_node(node = DeltaFn, type = Deterministic, interfaces = [:out, :in...], static_inputs = :fold)

    struct Linearization <: AbstractAlgorithm end
    @define_dependencies(
        node = DeltaFn,
        algorithm = Linearization,
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

    struct Structured <: AbstractAlgorithm end
    struct Gauss end
    @define_factor_node(node = Gauss, type = Stochastic, interfaces = [:out, :μ, :τ])
    @define_dependencies(
        node = Gauss,
        algorithm = Structured,
        dependencies = [:τ => (q[:out, :μ],), :out => (m[:μ], q[:τ])],
        free_energy_partition = [(:out, :μ), (:τ,)],
    )
end

@testitem "dependencies:declared" tags = [:base] setup = [DependencyNodes] begin
    using MessagePassingRulesBase: dependencies_spec, Target, IndexedTarget, VMP, BP, target_dependencies,
        Dependency, AllGroupMembers, AlignedGroupMember, AllGroupMembersButSelf, SingleInterface, CustomGroupSelector, static_inputs, free_energy_partition
    D = DependencyNodes

    # The node's own declaration applies to its default algorithm.
    declaration = dependencies_spec(D.NormalMixture, VMP())
    @test declaration !== nothing
    m_k = target_dependencies(declaration, IndexedTarget(:m, 2))
    @test map(d -> (d.container, d.key), m_k) == ((:q, :out), (:q, :switch), (:q, :p))
    @test m_k[3].selector isa AlignedGroupMember

    # Another algorithm, another declaration; an undeclared algorithm has none.
    delta = dependencies_spec(D.DeltaFn, D.Linearization())
    @test only(target_dependencies(delta, Target(:out))).selector isa AllGroupMembers
    @test target_dependencies(delta, IndexedTarget(:in, 1))[2].selector isa AllGroupMembersButSelf
    @test dependencies_spec(D.DeltaFn, BP()) === nothing
    @test target_dependencies(delta, Target(:nope)) === nothing

    @test static_inputs(D.DeltaFn) === :fold
    @test static_inputs(D.NormalMixture) === :none

    # Consumed and scored are separate: `τ` consumes the joint, the free_energy_partition says what is scored.
    structured = dependencies_spec(D.Gauss, D.Structured())
    τ = only(target_dependencies(structured, Target(:τ)))
    @test τ.key === (:out, :μ)
    @test free_energy_partition(structured) === ((:out, :μ), (:τ,))
    @test free_energy_partition(declaration) === nothing
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
    @test failure(node(:([:out => (m[:μ], m[:μ])]))) |> msg -> contains(msg, "twice")
    @test failure(node(:([:out => (m[:μ],), :out => (q[:μ],)]))) |> msg -> contains(msg, "declared twice")
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
