export DeltaFn

struct DeltaFnCallableWrapper{F} end

(::Type{DeltaFnCallableWrapper{F}})(args...) where {F} = F.instance(args...)

struct DeltaFn{F} end
struct DeltaFnNode{F, N, L, M} <: AbstractFactorNode
    fn::F

    out::NodeInterface
    ins::NTuple{N, IndexedNodeInterface}

    localmarginals :: L
    metadata       :: M
end

# include approximations
include("approximations/linearization.jl")
include("approximations/sampling.jl")

as_node_symbol(::Type{DeltaFn{ReactiveMP.DeltaFnCallableWrapper{F}}}) where {F} = Symbol(:DeltaFn, string(F))

functionalform(factornode::DeltaFnNode{F}) where {F}      = DeltaFn{DeltaFnCallableWrapper{F}}
sdtype(factornode::DeltaFnNode)                           = Deterministic()
interfaces(factornode::DeltaFnNode)                       = (factornode.out, factornode.ins...)
factorisation(factornode::DeltaFnNode{F, N}) where {F, N} = ntuple(identity, N + 1)
localmarginals(factornode::DeltaFnNode)                   = factornode.localmarginals.marginals
localmarginalnames(factornode::DeltaFnNode)               = map(name, localmarginals(factornode))
metadata(factornode::DeltaFnNode)                         = factornode.metadata

function __make_delta_fn_node(
    fn::F,
    options::FactorNodeCreationOptions,
    out::AbstractVariable,
    ins::NTuple{N, <:AbstractVariable}
) where {F <: Function, N, M}
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
    meta           = collect_meta(DeltaFn, metadata(options))

    return DeltaFnNode(fn, out_interface, ins_interface, localmarginals, meta)
end

function make_node(
    fform::F,
    options::FactorNodeCreationOptions,
    autovar::AutoVar,
    args::Vararg{<:AbstractVariable}
) where {F <: Function}
    out = randomvar(getname(autovar))
    return __make_delta_fn_node(fform, options, out, args), out
end

function make_node(fform::F, options::FactorNodeCreationOptions, args::Vararg{<:AbstractVariable}) where {F <: Function}
    return __make_delta_fn_node(fform, options, args[1], args[2:end])
end

# DeltaFn is very special, so it has a very special `activate!` function
function activate!(model, factornode::DeltaFnNode)
    fform = functionalform(factornode)
    meta  = metadata(factornode)

    foreach(interfaces(factornode)) do interface
        (connectedvar(interface) !== nothing) || error("Empty variable on interface $(interface) of node $(factornode)")
    end

    # First we declare local marginal clusters 
    let out = factornode.out, localmarginal = factornode.localmarginals.marginals[1]
        setstream!(localmarginal, getmarginal(connectedvar(out), IncludeAll()))
    end

    let out = factornode.out, ins = factornode.ins, localmarginal = factornode.localmarginals.marginals[2]
        cmarginal = MarginalObservable()
        setstream!(localmarginal, cmarginal)

        msgs_names      = Val{(:ins,)}
        msgs_observable = combineLatestUpdates((combineLatestUpdates(map((in) -> messagein(in), ins), PushNew()),), PushNew())

        marginal_names       = Val{(:out,)}
        marginals_observable = combineLatestUpdates((getmarginal(connectedvar(out), IncludeAll()),), PushNew())

        fform = functionalform(factornode)
        vtag  = Val{:ins}
        meta  = metadata(factornode)

        mapping     = MarginalMapping(fform, vtag, msgs_names, marginal_names, meta, factornode)
        marginalout = combineLatest((msgs_observable, marginals_observable), PushNew()) |> discontinue() |> map(Marginal, mapping)

        connect!(cmarginal, marginalout) # MarginalObservable has RecentSubject by default, there is no need to share_recent() here
    end

    # Second we declare message passing logic for out interface
    let interface = factornode.out
        msgs_names      = Val{(:out, :ins)}
        msgs_observable = combineLatestUpdates((messagein(interface), combineLatestUpdates(map((in) -> messagein(in), factornode.ins), PushNew())), PushNew())

        marginal_names       = nothing
        marginals_observable = of(nothing)

        vtag        = tag(interface)
        vconstraint = local_constraint(interface)

        vmessageout = combineLatest((msgs_observable, marginals_observable), PushNew())
        # vmessageout = apply_pipeline_stage(get_pipeline_stages(interface), factornode, vtag, vmessageout)

        mapping =
            let messagemap = MessageMapping(fform, vtag, vconstraint, msgs_names, marginal_names, meta, factornode)
                (dependencies) -> VariationalMessage(dependencies[1], dependencies[2], messagemap)
            end

        vmessageout = vmessageout |> map(AbstractMessage, mapping)
        vmessageout = apply_pipeline_stage(get_pipeline_stages(getoptions(model)), factornode, vtag, vmessageout)
        # vmessageout = apply_pipeline_stage(node_pipeline_extra_stages, factornode, vtag, vmessageout)
        vmessageout = vmessageout |> schedule_on(global_reactive_scheduler(getoptions(model)))

        connect!(messageout(interface), vmessageout)
    end

    # At last we declare message passing logic for input interfaces
    foreach(factornode.ins) do interface
        msgs_names      = Val{(:in,)}
        msgs_observable = combineLatestUpdates((messagein(interface),), PushNew())

        marginal_names       = Val{(:ins,)}
        marginals_observable = combineLatestUpdates((getstream(factornode.localmarginals.marginals[2]),), PushNew())

        vtag        = tag(interface)
        vconstraint = local_constraint(interface)

        vmessageout = combineLatest((msgs_observable, marginals_observable), PushNew())
        # vmessageout = apply_pipeline_stage(get_pipeline_stages(interface), factornode, vtag, vmessageout)

        mapping =
            let messagemap = MessageMapping(fform, vtag, vconstraint, msgs_names, marginal_names, meta, factornode)
                (dependencies) -> VariationalMessage(dependencies[1], dependencies[2], messagemap)
            end

        vmessageout = vmessageout |> map(AbstractMessage, mapping)
        vmessageout = apply_pipeline_stage(get_pipeline_stages(getoptions(model)), factornode, vtag, vmessageout)
        # vmessageout = apply_pipeline_stage(node_pipeline_extra_stages, factornode, vtag, vmessageout)
        vmessageout = vmessageout |> schedule_on(global_reactive_scheduler(getoptions(model)))

        connect!(messageout(interface), vmessageout)
    end
end