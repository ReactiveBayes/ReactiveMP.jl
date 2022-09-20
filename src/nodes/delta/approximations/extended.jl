using Parameters

export DeltaExtended, ET

@with_kw struct DeltaExtended{T}
    inverse::T = nothing
end

const ET = DeltaExtended

# DeltaFn node non-standard rule layout for `DeltaExtended` approximation rule
# `DeltaExtended` node changes the default layout in 2 ways:
# - `m_out` requires only inbounds messages on `m_in`s (TODO consider exchanging this behaviour with CVI in the future)
# - (ONLY WITH KNOWN INVERSE) `m_in(k)` requires only inbound message on `m_out` and inbound messages on the remaining `m_in`s

struct DeltaExtendedUknownInverseApproximationDeltaFnRuleLayout end

deltafn_rule_layout(::DeltaFnNode, ::DeltaExtended{Nothing}) =
    DeltaExtendedUknownInverseApproximationDeltaFnRuleLayout()

deltafn_apply_layout(
    ::DeltaExtendedUknownInverseApproximationDeltaFnRuleLayout,
    ::Val{:q_out},
    model,
    factornode::DeltaFnNode
) =
    deltafn_apply_layout(DeltaFnDefaultRuleLayout(), Val(:q_out), model, factornode)

deltafn_apply_layout(
    ::DeltaExtendedUknownInverseApproximationDeltaFnRuleLayout,
    ::Val{:q_ins},
    model,
    factornode::DeltaFnNode
) =
    deltafn_apply_layout(DeltaFnDefaultRuleLayout(), Val(:q_ins), model, factornode)

deltafn_apply_layout(
    ::DeltaExtendedUknownInverseApproximationDeltaFnRuleLayout,
    ::Val{:m_in},
    model,
    factornode::DeltaFnNode
) =
    deltafn_apply_layout(DeltaFnDefaultRuleLayout(), Val(:m_in), model, factornode)

function deltafn_apply_layout(
    ::DeltaExtendedUknownInverseApproximationDeltaFnRuleLayout,
    ::Val{:m_out},
    model,
    factornode::DeltaFnNode
)
    let out = factornode.out, ins = factornode.ins
        msgs_names      = Val{(:ins,)}
        msgs_observable = combineLatestUpdates((combineLatestMessagesInUpdates(ins), ), PushNew())

        # By default we don't need any marginals
        marginal_names       = nothing
        marginals_observable = of(nothing)

        fform       = functionalform(factornode)
        vtag        = tag(out)
        vconstraint = local_constraint(out)
        meta        = metadata(factornode)

        vmessageout = combineLatest((msgs_observable, marginals_observable), PushNew())
        # TODO
        # vmessageout = apply_pipeline_stage(get_pipeline_stages(interface), factornode, vtag, vmessageout)

        mapping =
            let messagemap = MessageMapping(fform, vtag, vconstraint, msgs_names, marginal_names, meta, factornode)
                (dependencies) -> VariationalMessage(dependencies[1], dependencies[2], messagemap)
            end

        vmessageout = vmessageout |> map(AbstractMessage, mapping)
        vmessageout = apply_pipeline_stage(get_pipeline_stages(getoptions(model)), factornode, vtag, vmessageout)
        # TODO
        # vmessageout = apply_pipeline_stage(node_pipeline_extra_stages, factornode, vtag, vmessageout)
        vmessageout = vmessageout |> schedule_on(global_reactive_scheduler(getoptions(model)))

        connect!(messageout(out), vmessageout)
    end
end

# We use non-standard `DeltaFn` node layout in case if inverse is known
struct DeltaExtendedKnownInverseApproximationDeltaFnRuleLayout end

deltafn_rule_layout(::DeltaFnNode, ::DeltaExtended{F}) where {F <: Function}               = DeltaExtendedKnownInverseApproximationDeltaFnRuleLayout()
deltafn_rule_layout(::DeltaFnNode, ::DeltaExtended{F}) where {N, F <: NTuple{N, Function}} = DeltaExtendedKnownInverseApproximationDeltaFnRuleLayout()

deltafn_apply_layout(
    ::DeltaExtendedKnownInverseApproximationDeltaFnRuleLayout,
    ::Val{:q_out},
    model,
    factornode::DeltaFnNode
) =
    deltafn_apply_layout(DeltaExtendedUknownInverseApproximationDeltaFnRuleLayout(), Val(:q_out), model, factornode)

deltafn_apply_layout(
    ::DeltaExtendedKnownInverseApproximationDeltaFnRuleLayout,
    ::Val{:q_ins},
    model,
    factornode::DeltaFnNode
) =
    deltafn_apply_layout(DeltaExtendedUknownInverseApproximationDeltaFnRuleLayout(), Val(:q_ins), model, factornode)

deltafn_apply_layout(
    ::DeltaExtendedKnownInverseApproximationDeltaFnRuleLayout,
    ::Val{:m_out},
    model,
    factornode::DeltaFnNode
) =
    deltafn_apply_layout(DeltaExtendedUknownInverseApproximationDeltaFnRuleLayout(), Val(:m_out), model, factornode)

# This function declares how to compute `m_in` 

function deltafn_apply_layout(
    ::DeltaExtendedKnownInverseApproximationDeltaFnRuleLayout,
    ::Val{:m_in},
    model,
    factornode::DeltaFnNode{F, N}
) where {F, N}
    # For each outbound message from `in_k` edge we need an inbound messages from all OTHER! `in_*` edges and inbound message on `m_out`
    foreach(enumerate(factornode.ins)) do (index, interface)

        # If we have only one `interface` we replace it with nothing
        # In other cases we remove the current index from the list of interfaces
        msgs_ins_stream = if N === 1 # `N` should be known at compile-time here so this `if` branch must be compiled out
            of(Message(nothing, true, true))
        else
            combineLatestMessagesInUpdates(TupleTools.deleteat(factornode.ins, index))
        end

        msgs_names      = Val{(:out, :ins)}
        msgs_observable = combineLatestUpdates((messagein(factornode.out), msgs_ins_stream), PushNew())

        marginal_names       = nothing
        marginals_observable = of(nothing)

        fform       = functionalform(factornode)
        vtag        = tag(interface)
        vconstraint = local_constraint(interface)
        meta        = metadata(factornode)

        vmessageout = combineLatest((msgs_observable, marginals_observable), PushNew())
        # vmessageout = apply_pipeline_stage(get_pipeline_stages(interface), factornode, vtag, vmessageout)

        mapping =
            let messagemap = MessageMapping(fform, vtag, vconstraint, msgs_names, marginal_names, meta, factornode)
                (dependencies) -> VariationalMessage(dependencies[1], dependencies[2], messagemap)
            end

        vmessageout = vmessageout |> map(AbstractMessage, mapping)
        vmessageout = apply_pipeline_stage(get_pipeline_stages(getoptions(model)), factornode, vtag, vmessageout)
        # TODO
        # vmessageout = apply_pipeline_stage(node_pipeline_extra_stages, factornode, vtag, vmessageout)
        vmessageout = vmessageout |> schedule_on(global_reactive_scheduler(getoptions(model)))

        connect!(messageout(interface), vmessageout)
    end
end
