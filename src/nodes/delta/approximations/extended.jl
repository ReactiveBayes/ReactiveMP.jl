export DeltaExtended, ET

struct DeltaExtended{T}
    inverse::T
end

DeltaExtended() = DeltaExtended(nothing)

const ET = DeltaExtended


# DeltaFn node non-standard rule layout for `DeltaExtended` approximation rule

struct DeltaExtendednUknownInverseApproximationDeltaFnRuleLayout end

deltafn_rule_layout(::DeltaFnNode, ::DeltaExtended{Nothing}) = DeltaExtendednUknownInverseApproximationDeltaFnRuleLayout()

deltafn_apply_layout(::DeltaExtendednUknownInverseApproximationDeltaFnRuleLayout, ::Val{:q_out}, model, factornode::DeltaFnNode) =
    deltafn_apply_layout(DeltaFnDefaultRuleLayout(), Val(:q_out), model, factornode)

deltafn_apply_layout(::DeltaExtendednUknownInverseApproximationDeltaFnRuleLayout, ::Val{:q_ins}, model, factornode::DeltaFnNode) =
    deltafn_apply_layout(DeltaFnDefaultRuleLayout(), Val(:q_ins), model, factornode)

deltafn_apply_layout(::DeltaExtendednUknownInverseApproximationDeltaFnRuleLayout, ::Val{:m_in}, model, factornode::DeltaFnNode) =
    deltafn_apply_layout(DeltaFnDefaultRuleLayout(), Val(:m_in), model, factornode)

function deltafn_apply_layout(::DeltaExtendednUknownInverseApproximationDeltaFnRuleLayout, ::Val{:m_out}, model, factornode::DeltaFnNode)
    let out = factornode.out, ins = factornode.ins
        # By default, to compute an outbound message on `:out` edge we need inbound messages both from `:ins` edge
        msgs_names      = Val{(:ins,)}
        msgs_observable = combineLatestUpdates(
            (combineLatestUpdates(map((in) -> messagein(in), ins), PushNew()), ), 
        PushNew())

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

##

struct DeltaExtendedKnownInverseApproximationDeltaFnRuleLayout end

# We use non-standard `DeltaFn` node layout in case if inverse is known
deltafn_rule_layout(::DeltaFnNode, ::DeltaExtended{F}) where { F <: Function } = DeltaExtendedKnownInverseApproximationDeltaFnRuleLayout()

deltafn_apply_layout(::DeltaExtendedKnownInverseApproximationDeltaFnRuleLayout, ::Val{:q_out}, model, factornode::DeltaFnNode) =
    deltafn_apply_layout(DeltaExtendednUknownInverseApproximationDeltaFnRuleLayout(), Val(:q_out), model, factornode)

deltafn_apply_layout(::DeltaExtendedKnownInverseApproximationDeltaFnRuleLayout, ::Val{:q_ins}, model, factornode::DeltaFnNode) =
    deltafn_apply_layout(DeltaExtendednUknownInverseApproximationDeltaFnRuleLayout(), Val(:q_ins), model, factornode)

deltafn_apply_layout(::DeltaExtendedKnownInverseApproximationDeltaFnRuleLayout, ::Val{:m_out}, model, factornode::DeltaFnNode) =
    deltafn_apply_layout(DeltaExtendednUknownInverseApproximationDeltaFnRuleLayout(), Val(:m_out), model, factornode)

# This function declares how to compute `m_in` 

function deltafn_apply_layout(::DeltaExtendedKnownInverseApproximationDeltaFnRuleLayout, ::Val{:m_in}, model, factornode::DeltaFnNode) 
    # For each outbound message from `in_k` edge we need an inbound messages from all OTHER! `in_*` edges and inbound message on `m_out`
    foreach(enumerate(factornode.ins)) do (index, interface)

        msgs_without_current = TupleTools.deleteat(factornode.ins, index)
        msgs_names           = Val{(:out, :ins,)}
        msgs_ins_stream      = !isempty(msgs_without_current) ? combineLatestUpdates(map((in) -> messagein(in), msgs_without_current), PushNew()) : of(nothing)
        msgs_observable      = combineLatestUpdates((messagein(factornode.out), msgs_ins_stream,), PushNew())

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