
using Rocket
import ReactiveMP:
    deltafn_rule_layout,
    deltafn_apply_layout,
    AbstractDeltaNodeDependenciesLayout,
    DeltaFnDefaultRuleLayout,
    DeltaFnNode,
    getmarginal,
    functionalform,
    tag,
    Marginalisation,
    MessageMapping,
    DeferredMessage,
    with_statics,
    apply_pipeline_stage,
    messageout,
    messagein,
    connect!

"""
    CVIProjectionApproximationDeltaFnRuleLayout

Custom rule layout for the Delta node in case of the CVI projection approximation method:

# Layout 

In order to compute:

- `q_out`: mirrors the posterior marginal on the `out` edge
- `q_ins`: uses inbound message on the `out` edge and all inbound messages on the `ins` edges
- `m_out`: uses the posterior over `out`, message from `out` and the joint over the `ins` edges
- `m_in_k`: uses the inbound message on the `in_k` edge and `q_ins`
"""
struct CVIProjectionApproximationDeltaFnRuleLayout <: AbstractDeltaNodeDependenciesLayout end

deltafn_rule_layout(::DeltaFnNode, ::CVIProjection, inverse::Nothing) = CVIProjectionApproximationDeltaFnRuleLayout()

function deltafn_rule_layout(::DeltaFnNode, ::CVIProjection, inverse::Any)
    @warn "CVI projection approximation does not accept the inverse function. Ignoring the provided inverse."
    return CVIProjectionApproximationDeltaFnRuleLayout()
end

# This function declares how to compute `q_out` locally around `DeltaFn`
function deltafn_apply_layout(::CVIProjectionApproximationDeltaFnRuleLayout, ::Val{:q_out}, factornode::DeltaFnNode, meta, pipeline_stages, scheduler, addons, rulefallback)
    return deltafn_apply_layout(DeltaFnDefaultRuleLayout(), Val(:q_out), factornode, meta, pipeline_stages, scheduler, addons, rulefallback)
end

# This function declares how to compute `q_ins` locally around `DeltaFn`
function deltafn_apply_layout(::CVIProjectionApproximationDeltaFnRuleLayout, ::Val{:q_ins}, factornode::DeltaFnNode, meta, pipeline_stages, scheduler, addons, rulefallback)
    return deltafn_apply_layout(DeltaFnDefaultRuleLayout(), Val(:q_ins), factornode, meta, pipeline_stages, scheduler, addons, rulefallback)
end

# This function declares how to compute `m_out` 
function deltafn_apply_layout(::CVIProjectionApproximationDeltaFnRuleLayout, ::Val{:m_out}, factornode::DeltaFnNode, meta, pipeline_stages, scheduler, addons, rulefallback)
    let interface = factornode.out
        msgs_names      = Val{(:out,)}()
        msgs_observable = combineLatestUpdates((messagein(factornode.out),), PushNew())

        marginal_names       = Val{(:out, :ins)}()
        marginals_observable = combineLatestUpdates((getmarginal(factornode.localmarginals.marginals[1]), getmarginal(factornode.localmarginals.marginals[2])), PushNew())

        fform       = functionalform(factornode)
        vtag        = tag(interface)
        vconstraint = Marginalisation()

        vmessageout = combineLatest((msgs_observable, marginals_observable), PushNew())

        mapping = let messagemap = MessageMapping(fform, vtag, vconstraint, msgs_names, marginal_names, meta, addons, factornode, rulefallback)
            (dependencies) -> DeferredMessage(dependencies[1], dependencies[2], messagemap)
        end

        vmessageout = with_statics(factornode, vmessageout)
        vmessageout = vmessageout |> map(AbstractMessage, mapping)
        vmessageout = apply_pipeline_stage(pipeline_stages, factornode, vtag, vmessageout)
        vmessageout = vmessageout |> schedule_on(scheduler)

        connect!(messageout(interface), vmessageout)
    end
end

# This function declares how to compute `m_in` for each `k` 
function deltafn_apply_layout(::CVIProjectionApproximationDeltaFnRuleLayout, ::Val{:m_in}, factornode::DeltaFnNode, meta, pipeline_stages, scheduler, addons, rulefallback)
    return deltafn_apply_layout(DeltaFnDefaultRuleLayout(), Val(:m_in), factornode, meta, pipeline_stages, scheduler, addons, rulefallback)
end
