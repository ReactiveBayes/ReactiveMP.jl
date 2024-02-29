
"""
    DeltaFnDefaultRuleLayout

Default rule layout for the Delta node:

# Layout 

In order to compute:

- `q_out`: mirrors the posterior marginal on the `out` edge
- `q_ins`: uses inbound message on the `out` edge and all inbound messages on the `ins` edges
- `m_out`: uses all inbound messages on the `ins` edges
- `m_in_k`: uses the inbound message on the `in_k` edge and `q_ins`

See also: [`ReactiveMP.DeltaFnDefaultKnownInverseRuleLayout`](@ref)
"""
struct DeltaFnDefaultRuleLayout <: AbstractDeltaNodeDependenciesLayout end

import FixedArguments

function with_statics(factornode::DeltaFnNode, stream)
    return with_statics(factornode, factornode.statics, stream)
end

function with_statics(factornode::DeltaFnNode, statics::Tuple, stream::T) where {T}
    # We wait for the statics to be available, but ignore their actual values 
    # They are being injected indirectly with the `fix` function upon node creation
    statics = map(static -> messageout(static, 1), FixedArguments.value.(factornode.statics))
    return combineLatest((stream, combineLatest(statics, PushNew()))) |> map(eltype(T), first)
end

function with_statics(factornode::DeltaFnNode, statics::Tuple{}, stream::T) where {T}
    # There is no need to touch the original stream if there are no statics
    return stream
end

# This function declares how to compute `q_out` locally around `DeltaFn`
function deltafn_apply_layout(::DeltaFnDefaultRuleLayout, ::Val{:q_out}, factornode::DeltaFnNode, pipeline_stages, scheduler, addons)
    let out = factornode.out, localmarginal = factornode.localmarginals.marginals[1]
        # We simply subscribe on the marginal of the connected variable on `out` edge
        setstream!(localmarginal, getmarginal(connected_properties(out), IncludeAll()))
    end
end

# This function declares how to compute `q_ins` locally around `DeltaFn`
function deltafn_apply_layout(::DeltaFnDefaultRuleLayout, ::Val{:q_ins}, factornode::DeltaFnNode, pipeline_stages, scheduler, addons)
    let out = factornode.out, ins = factornode.ins, localmarginal = factornode.localmarginals.marginals[2]
        cmarginal = MarginalObservable()
        setstream!(localmarginal, cmarginal)

        # By default to compute `q_ins` we need messages both from `:out` and `:ins`
        msgs_names      = Val{(:out, :ins)}()
        msgs_observable = combineLatestUpdates((messagein(out), combineLatestMessagesInUpdates(ins)), PushNew())

        # By default, we should not need any local marginals
        marginal_names       = nothing
        marginals_observable = of(nothing)

        fform = functionalform(factornode)
        vtag  = Val{:ins}()
        meta  = metadata(factornode)

        mapping     = MarginalMapping(fform, vtag, msgs_names, marginal_names, meta, factornode)
        marginalout = with_statics(factornode, combineLatest((msgs_observable, marginals_observable), PushNew())) |> discontinue() |> map(Marginal, mapping)

        connect!(cmarginal, marginalout) # MarginalObservable has RecentSubject by default, there is no need to share_recent() here
    end
end

# This function declares how to compute `m_out` 
function deltafn_apply_layout(::DeltaFnDefaultRuleLayout, ::Val{:m_out}, factornode::DeltaFnNode, pipeline_stages, scheduler, addons)
    let out = factornode.out, ins = factornode.ins
        node_pipeline              = getpipeline(factornode)
        node_pipeline_extra_stages = get_pipeline_stages(node_pipeline)

        # By default we simply request all inbound messages from `ins` edges
        msgs_names      = Val{(:ins,)}()
        msgs_observable = combineLatestUpdates((combineLatestMessagesInUpdates(ins),), PushNew())

        # By default we don't need any marginals
        marginal_names       = nothing
        marginals_observable = of(nothing)

        fform       = functionalform(factornode)
        vtag        = Val{:out}()
        vconstraint = local_constraint(out)
        meta        = metadata(factornode)

        vmessageout = combineLatest((msgs_observable, marginals_observable), PushNew())

        mapping = let messagemap = MessageMapping(fform, vtag, vconstraint, msgs_names, marginal_names, meta, addons, factornode)
            (dependencies) -> VariationalMessage(dependencies[1], dependencies[2], messagemap)
        end

        vmessageout = with_statics(factornode, vmessageout)
        vmessageout = vmessageout |> map(AbstractMessage, mapping)
        vmessageout = apply_pipeline_stage(pipeline_stages, factornode, vtag, vmessageout)
        vmessageout = apply_pipeline_stage(node_pipeline_extra_stages, factornode, vtag, vmessageout)
        vmessageout = vmessageout |> schedule_on(scheduler)

        connect!(messageout(out), vmessageout)
    end
end

# This function declares how to compute `m_in` for each `k` 
function deltafn_apply_layout(::DeltaFnDefaultRuleLayout, ::Val{:m_in}, factornode::DeltaFnNode, pipeline_stages, scheduler, addons)
    node_pipeline              = getpipeline(factornode)
    node_pipeline_extra_stages = get_pipeline_stages(node_pipeline)

    # For each outbound message from `in_k` edge we need an inbound message on this edge and a joint marginal over `:ins` edges
    foreach(factornode.ins) do interface
        msgs_names      = Val{(:in,)}()
        msgs_observable = combineLatestUpdates((messagein(interface),), PushNew())

        marginal_names       = Val{(:ins,)}()
        marginals_observable = combineLatestUpdates((getstream(factornode.localmarginals.marginals[2]),), PushNew())

        fform       = functionalform(factornode)
        vtag        = tag(interface)
        vconstraint = local_constraint(interface)
        meta        = metadata(factornode)

        vmessageout = combineLatest((msgs_observable, marginals_observable), PushNew())

        mapping = let messagemap = MessageMapping(fform, vtag, vconstraint, msgs_names, marginal_names, meta, addons, factornode)
            (dependencies) -> VariationalMessage(dependencies[1], dependencies[2], messagemap)
        end

        vmessageout = with_statics(factornode, vmessageout)
        vmessageout = vmessageout |> map(AbstractMessage, mapping)
        vmessageout = apply_pipeline_stage(pipeline_stages, factornode, vtag, vmessageout)
        vmessageout = apply_pipeline_stage(node_pipeline_extra_stages, factornode, vtag, vmessageout)
        vmessageout = vmessageout |> schedule_on(scheduler)

        connect!(messageout(interface), vmessageout)
    end
end

"""
    DeltaFnDefaultKnownInverseRuleLayout

Default rule layout for the Delta node:

# Layout 

In order to compute:

- `q_out`: mirrors the posterior marginal on the `out` edge (same as the `DeltaFnDefaultRuleLayout`)
- `q_ins`: uses inbound message on the `out` edge and all inbound messages on the `ins` edges (same as the `DeltaFnDefaultRuleLayout`)
- `m_out`: uses all inbound messages on the `ins` edges (same as the `DeltaFnDefaultRuleLayout`)
- `m_in_k`: uses inbound message on the `out` edge and inbound messages on the `ins` edges except `k`

See also: [`ReactiveMP.DeltaFnDefaultRuleLayout`](@ref)
"""
struct DeltaFnDefaultKnownInverseRuleLayout <: AbstractDeltaNodeDependenciesLayout end

function deltafn_apply_layout(::DeltaFnDefaultKnownInverseRuleLayout, ::Val{:q_out}, factornode::DeltaFnNode, pipeline_stages, scheduler, addons)
    return deltafn_apply_layout(DeltaFnDefaultRuleLayout(), Val(:q_out), factornode, pipeline_stages, scheduler, addons)
end

function deltafn_apply_layout(::DeltaFnDefaultKnownInverseRuleLayout, ::Val{:q_ins}, factornode::DeltaFnNode, pipeline_stages, scheduler, addons)
    return deltafn_apply_layout(DeltaFnDefaultRuleLayout(), Val(:q_ins), factornode, pipeline_stages, scheduler, addons)
end

function deltafn_apply_layout(::DeltaFnDefaultKnownInverseRuleLayout, ::Val{:m_out}, factornode::DeltaFnNode, pipeline_stages, scheduler, addons)
    return deltafn_apply_layout(DeltaFnDefaultRuleLayout(), Val(:m_out), factornode, pipeline_stages, scheduler, addons)
end

# This function declares how to compute `m_in` 
function deltafn_apply_layout(::DeltaFnDefaultKnownInverseRuleLayout, ::Val{:m_in}, factornode::DeltaFnNode{F}, pipeline_stages, scheduler, addons) where {F}
    N = length(factornode.ins)

    node_pipeline              = getpipeline(factornode)
    node_pipeline_extra_stages = get_pipeline_stages(node_pipeline)

    # For each outbound message from `in_k` edge we need an inbound messages from all OTHER! `in_*` edges and inbound message on `m_out`
    foreach(enumerate(factornode.ins)) do (index, interface)

        # If we have only one `interface` we replace it with nothing
        # In other cases we remove the current index from the list of interfaces
        msgs_ins_stream = if N === 1 # `N` should be known at compile-time here so this `if` branch must be compiled out
            of(Message(nothing, true, true, nothing))
        else
            combineLatestMessagesInUpdates(TupleTools.deleteat(factornode.ins, index))
        end

        msgs_names      = Val{(:out, :ins)}()
        msgs_observable = combineLatestUpdates((messagein(factornode.out), msgs_ins_stream), PushNew())

        marginal_names       = nothing
        marginals_observable = of(nothing)

        fform       = functionalform(factornode)
        vtag        = tag(interface)
        vconstraint = local_constraint(interface)
        meta        = metadata(factornode)

        vmessageout = combineLatest((msgs_observable, marginals_observable), PushNew())

        # TODO add addons
        mapping = let messagemap = MessageMapping(fform, vtag, vconstraint, msgs_names, marginal_names, meta, addons, factornode)
            (dependencies) -> VariationalMessage(dependencies[1], dependencies[2], messagemap)
        end

        vmessageout = with_statics(factornode, vmessageout)
        vmessageout = vmessageout |> map(AbstractMessage, mapping)
        vmessageout = apply_pipeline_stage(pipeline_stages, factornode, vtag, vmessageout)
        vmessageout = apply_pipeline_stage(node_pipeline_extra_stages, factornode, vtag, vmessageout)
        vmessageout = vmessageout |> schedule_on(scheduler)

        connect!(messageout(interface), vmessageout)
    end
end
