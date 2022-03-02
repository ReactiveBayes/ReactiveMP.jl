
struct DeltaFn end

struct DeltaFnNode{F, N, L, M} <: AbstractFactorNode
    fn :: F
    
    out :: NodeInterface
    ins :: NTuple{N, IndexedNodeInterface}
    
    localmarginals :: L
    metadata       :: M
end

functionalform(factornode::DeltaFnNode)                     = DeltaFn
sdtype(factornode::DeltaFnNode)                             = Deterministic()
interfaces(factornode::DeltaFnNode)                         = (factornode.out, factornode.ins...)
factorisation(factornode::DeltaFnNode{F, N}) where { F, N } = ntuple(identity, N + 1)
localmarginals(factornode::DeltaFnNode)                     = factornode.localmarginals.marginals
localmarginalnames(factornode::DeltaFnNode)                 = map(name, localmarginals(factornode))
metadata(factornode::DeltaFnNode)                           = factornode.metadata

function __make_delta_fn_node(fn::F, out::AbstractVariable, ins::NTuple{N, <: AbstractVariable}; factorisation = nothing, meta::M = nothing) where { F <: Function, N, M }
    out_interface = NodeInterface(:out, Marginalisation())
    ins_interface = ntuple(i -> IndexedNodeInterface(i, NodeInterface(:in, Marginalisation())), N)

    out_index = getlastindex(out)
    connectvariable!(out_interface, out, out_index)
    setmessagein!(out, out_index, messageout(out_interface))

    foreach(zip(ins_interface, ins)) do (in_interface, in_var)
        in_index = getlastindex(in_var)
        connectvariable!(in_interface, in_var, in_index)
        setmessagein!(in_var, in_index, messageout(in_interface))
    end

    localmarginals = FactorNodeLocalMarginals((FactorNodeLocalMarginal(1, :out), FactorNodeLocalMarginal(2, :ins)))
    metadata       = collect_meta(DeltaFn, meta)

    return DeltaFnNode(fn, out_interface, ins_interface, localmarginals, metadata)
end

function make_node(fform::F, autovar::AutoVar, args::Vararg{ <: AbstractVariable }; kwargs...) where { F <: Function }
    out = randomvar(getname(autovar))
    return __make_delta_fn_node(fform, out, args; kwargs...), out
end

function make_node(fform::F, args::Vararg{ <: AbstractVariable }; kwargs...) where { F <: Function }
    return __make_delta_fn_node(fform, args[1], args[2:end]; kwargs...)
end

# DeltaFn is very special, so it has a very special `activate!` function
function activate!(model, factornode::DeltaFnNode)
    fform = functionalform(factornode)
    meta  = metadata(factornode)

    foreach(interfaces(factornode)) do interface
        (connectedvar(interface) !== nothing) || error("Empty variable on interface $(interface) of node $(factornode)")
    end

    # Declare marginals

    # First we declare message passing logic for out interface
    let interface = factornode.out

        msgs_names      = Val{ (:ins, ) }
        msgs_observable = combineLatest((combineLatest(map((in) -> messagein(in), factornode.ins), PushNew()), ), PushNew())

        marginal_names       = nothing
        marginals_observable = of(nothing)

        vtag        = tag(interface)
        vconstraint = local_constraint(interface)

        vmessageout = combineLatest((msgs_observable, marginals_observable), PushNew())
        # vmessageout = apply_pipeline_stage(get_pipeline_stages(interface), factornode, vtag, vmessageout)

        mapping = MessageMapping(fform, vtag, vconstraint, msgs_names, marginal_names, meta, factornode)
        mapping = apply_mapping(msgs_observable, marginals_observable, mapping)

        vmessageout = vmessageout |> map(AbstractMessage, mapping)
        vmessageout = apply_pipeline_stage(get_pipeline_stages(getoptions(model)), factornode, vtag, vmessageout)
        # vmessageout = apply_pipeline_stage(node_pipeline_extra_stages, factornode, vtag, vmessageout)
        vmessageout = vmessageout |> schedule_on(global_reactive_scheduler(getoptions(model)))

        connect!(messageout(interface), vmessageout)
    end

    

    # for (iindex, interface) in enumerate(interfaces(factornode))
    #     cvariable = connectedvar(interface)
    #     if cvariable !== nothing
    #         message_dependencies, marginal_dependencies = functional_dependencies(node_pipeline_dependencies, factornode, iindex)

    #         msgs_names, msgs_observable          = get_messages_observable(factornode, message_dependencies)
    #         marginal_names, marginals_observable = get_marginals_observable(factornode, marginal_dependencies)

    #         vtag        = tag(interface)
    #         vconstraint = local_constraint(interface)
            
    #         vmessageout = combineLatest((msgs_observable, marginals_observable), PushNew()) # TODO check PushEach
    #         vmessageout = apply_pipeline_stage(get_pipeline_stages(interface), factornode, vtag, vmessageout)

    #         mapping = MessageMapping(fform, vtag, vconstraint, msgs_names, marginal_names, meta, factornode)
    #         mapping = apply_mapping(msgs_observable, marginals_observable, mapping)

    #         vmessageout = vmessageout |> map(AbstractMessage, mapping)
    #         vmessageout = apply_pipeline_stage(get_pipeline_stages(getoptions(model)), factornode, vtag, vmessageout)
    #         vmessageout = apply_pipeline_stage(node_pipeline_extra_stages, factornode, vtag, vmessageout)
    #         vmessageout = vmessageout |> schedule_on(global_reactive_scheduler(getoptions(model)))

    #         connect!(messageout(interface), vmessageout)
    #     else
    #         error("Empty variable on interface $(interface) of node $(factornode)")
    #     end
    # end
end