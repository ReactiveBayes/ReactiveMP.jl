using Parameters

export DeltaUnscented, UT

const default_alpha = 1e-3 # Default value for the spread parameter
const default_beta = 2.0
const default_kappa = 0.0

@with_kw struct DeltaUnscented{T}
    inverse::T = nothing
    alpha::Real = default_alpha
    beta::Real = default_beta
    kappa::Real = default_kappa
end

const UT = DeltaUnscented

#TODO: Copied from extended.jl

# DeltaFn node non-standard rule layout for `DeltaUnscented` approximation rule
# `DeltaUnscented` node changes the default layout in 2 ways:
# - `m_out` requires only inbounds messages on `m_in`s (TODO consider exchanging this behaviour with CVI in the future)
# - (ONLY WITH KNOWN INVERSE) `m_in(k)` requires only inbound message on `m_out` and inbound messages on the remaining `m_in`s

struct DeltaUnscentedUknownInverseApproximationDeltaFnRuleLayout end

deltafn_rule_layout(::DeltaFnNode, ::DeltaUnscented{Nothing}) =
    DeltaUnscentedUknownInverseApproximationDeltaFnRuleLayout()

deltafn_apply_layout(
    ::DeltaUnscentedUknownInverseApproximationDeltaFnRuleLayout,
    ::Val{:q_out},
    model,
    factornode::DeltaFnNode
) =
    deltafn_apply_layout(DeltaFnDefaultRuleLayout(), Val(:q_out), model, factornode)

deltafn_apply_layout(
    ::DeltaUnscentedUknownInverseApproximationDeltaFnRuleLayout,
    ::Val{:q_ins},
    model,
    factornode::DeltaFnNode
) =
    deltafn_apply_layout(DeltaFnDefaultRuleLayout(), Val(:q_ins), model, factornode)

deltafn_apply_layout(
    ::DeltaUnscentedUknownInverseApproximationDeltaFnRuleLayout,
    ::Val{:m_in},
    model,
    factornode::DeltaFnNode
) =
    deltafn_apply_layout(DeltaFnDefaultRuleLayout(), Val(:m_in), model, factornode)

function deltafn_apply_layout(
    ::DeltaUnscentedUknownInverseApproximationDeltaFnRuleLayout,
    ::Val{:m_out},
    model,
    factornode::DeltaFnNode
)
    let out = factornode.out, ins = factornode.ins
        msgs_names      = Val{(:ins,)}
        msgs_observable = combineLatestUpdates((combineLatestUpdates(map((in) -> messagein(in), ins), PushNew()),), PushNew())

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
struct DeltaUnscentedKnownInverseApproximationDeltaFnRuleLayout end

deltafn_rule_layout(::DeltaFnNode, ::DeltaUnscented{F}) where {F <: Function}               = DeltaUnscentedKnownInverseApproximationDeltaFnRuleLayout()
deltafn_rule_layout(::DeltaFnNode, ::DeltaUnscented{F}) where {N, F <: NTuple{N, Function}} = DeltaUnscentedKnownInverseApproximationDeltaFnRuleLayout()

deltafn_apply_layout(
    ::DeltaUnscentedKnownInverseApproximationDeltaFnRuleLayout,
    ::Val{:q_out},
    model,
    factornode::DeltaFnNode
) =
    deltafn_apply_layout(DeltaUnscentedUknownInverseApproximationDeltaFnRuleLayout(), Val(:q_out), model, factornode)

deltafn_apply_layout(
    ::DeltaUnscentedKnownInverseApproximationDeltaFnRuleLayout,
    ::Val{:q_ins},
    model,
    factornode::DeltaFnNode
) =
    deltafn_apply_layout(DeltaUnscentedUknownInverseApproximationDeltaFnRuleLayout(), Val(:q_ins), model, factornode)

deltafn_apply_layout(
    ::DeltaUnscentedKnownInverseApproximationDeltaFnRuleLayout,
    ::Val{:m_out},
    model,
    factornode::DeltaFnNode
) =
    deltafn_apply_layout(DeltaUnscentedUknownInverseApproximationDeltaFnRuleLayout(), Val(:m_out), model, factornode)

# This function declares how to compute `m_in` 

function deltafn_apply_layout(
    ::DeltaUnscentedKnownInverseApproximationDeltaFnRuleLayout,
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
            combineLatestUpdates(map((in) -> messagein(in), TupleTools.deleteat(factornode.ins, index)), PushNew())
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
