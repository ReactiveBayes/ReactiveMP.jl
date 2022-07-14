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
include("approximations/cvi.jl")

as_node_symbol(::Type{DeltaFn{ReactiveMP.DeltaFnCallableWrapper{F}}}) where {F} = Symbol(:DeltaFn, string(F))

functionalform(factornode::DeltaFnNode{F}) where {F}      = DeltaFn{DeltaFnCallableWrapper{F}}
sdtype(factornode::DeltaFnNode)                           = Deterministic()
interfaces(factornode::DeltaFnNode)                       = (factornode.out, factornode.ins...)
factorisation(factornode::DeltaFnNode{F, N}) where {F, N} = ntuple(identity, N + 1)
localmarginals(factornode::DeltaFnNode)                   = factornode.localmarginals.marginals
localmarginalnames(factornode::DeltaFnNode)               = map(name, localmarginals(factornode))
metadata(factornode::DeltaFnNode)                         = factornode.metadata

# For missing rules error msg
rule_method_error_extract_fform(f::Type{<:DeltaFn}) = "DeltaFn{f}"

function interfaceindex(factornode::DeltaFnNode, iname::Symbol)
    if iname === :out
        return 1
    elseif iname === :ins || iname === :in
        return 2
    else
        error("Unknown interface ':$(iname)' for nonlinear delta fn [ $(functionalform(factornode)) ] node")
    end
end

function __make_delta_fn_node(
    fn::F,
    options::FactorNodeCreationOptions,
    out::AbstractVariable,
    ins::NTuple{N, <:AbstractVariable}
) where {F <: Function, N}
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

    localmarginals = FactorNodeLocalMarginals((FactorNodeLocalMarginal(1, 1, :out), FactorNodeLocalMarginal(2, 2, :ins)))
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

##

struct DeltaFnDefaultRuleLayout end

# By default all `meta` objects fallback to the `DeltaFnDefaultRuleLayout`
# We, however, allow for specific approximation methods to override the default `DeltaFn` rule layout for better efficiency
deltafn_rule_layout(factornode::DeltaFnNode)       = deltafn_rule_layout(factornode, metadata(factornode))  
deltafn_rule_layout(factornode::DeltaFnNode, meta) = DeltaFnDefaultRuleLayout()

# This function declares how to compute `q_out` locally around `DeltaFn`
function deltafn_apply_layout(::DeltaFnDefaultRuleLayout, ::Val{ :q_out }, model, factornode::DeltaFnNode)
    let out = factornode.out, localmarginal = factornode.localmarginals.marginals[1]
        # We simply subscribe on the marginal of the connected variable on `out` edge
        setstream!(localmarginal, getmarginal(connectedvar(out), IncludeAll()))
    end
end

# This function declares how to compute `q_ins` locally around `DeltaFn`
function deltafn_apply_layout(::DeltaFnDefaultRuleLayout, ::Val{ :q_ins }, model, factornode::DeltaFnNode)
    let out = factornode.out, ins = factornode.ins, localmarginal = factornode.localmarginals.marginals[2]
        cmarginal = MarginalObservable()
        setstream!(localmarginal, cmarginal)

        # By default to compute `q_ins` we need messages both from `:out` and `:ins`
        msgs_names      = Val{(:out, :ins)}
        msgs_observable = combineLatestUpdates((messagein(out), combineLatestUpdates(map((in) -> messagein(in), ins), PushNew())), PushNew())

        # By default, we should not need any local marginals
        marginal_names       = nothing
        marginals_observable = of(nothing)

        fform = functionalform(factornode)
        vtag  = Val{:ins}
        meta  = metadata(factornode)

        mapping     = MarginalMapping(fform, vtag, msgs_names, marginal_names, meta, factornode)
        marginalout = combineLatest((msgs_observable, marginals_observable), PushNew()) |> discontinue() |> map(Marginal, mapping)

        connect!(cmarginal, marginalout) # MarginalObservable has RecentSubject by default, there is no need to share_recent() here
    end
end

# This function declares how to compute `m_out` 
function deltafn_apply_layout(::DeltaFnDefaultRuleLayout, ::Val{ :m_out }, model, factornode::DeltaFnNode)
    let interface = factornode.out
        # By default, to compute an outbound message on `:out` edge we need inbound messages both from `:ins` edge and `:out` edge
        msgs_names      = Val{(:out, :ins)}
        msgs_observable = combineLatestUpdates((messagein(interface), combineLatestUpdates(map((in) -> messagein(in), factornode.ins), PushNew())), PushNew())

        # By default we don't need any marginals
        marginal_names       = nothing
        marginals_observable = of(nothing)

        fform       = functionalform(factornode)
        vtag        = tag(interface)
        vconstraint = local_constraint(interface)
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

        connect!(messageout(interface), vmessageout)
    end
end

# This function declares how to compute `m_in` for each `k` 
function deltafn_apply_layout(::DeltaFnDefaultRuleLayout, ::Val{ :m_in }, model, factornode::DeltaFnNode)
    # For each outbound message from `in_k` edge we need an inbound message on this edge and a joint marginal over `:ins` edges
    foreach(factornode.ins) do interface
        msgs_names      = Val{(:in,)}
        msgs_observable = combineLatestUpdates((messagein(interface),), PushNew())

        marginal_names       = Val{(:ins,)}
        marginals_observable = combineLatestUpdates((getstream(factornode.localmarginals.marginals[2]),), PushNew())

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

##

function activate!(model, factornode::DeltaFnNode)
    # `DeltaFn` node may change rule arguments layout depending on the `meta`
    # This feature is similar to `functional_dependencies` for a regular `FactorNode` implementation
    return activate!(model, factornode, deltafn_rule_layout(factornode))
end

# DeltaFn is very special, so it has a very special `activate!` function
function activate!(model, factornode::DeltaFnNode, layout)
    foreach(interfaces(factornode)) do interface
        (connectedvar(interface) !== nothing) || error("Empty variable on interface $(interface) of node $(factornode)")
    end

    # First we declare local marginal for `out` edge
    deltafn_apply_layout(layout, Val(:q_out), model, factornode)

    # Second we declare how to compute a joint marginal over all inbound edges
    deltafn_apply_layout(layout, Val(:q_ins), model, factornode)

    # Second we declare message passing logic for out interface
    deltafn_apply_layout(layout, Val(:m_out), model, factornode)

    # At last we declare message passing logic for input interfaces
    deltafn_apply_layout(layout, Val(:m_in), model, factornode)
end
