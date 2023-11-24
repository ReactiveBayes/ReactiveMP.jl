export DeltaFn, DeltaFnNode, DeltaMeta

"""
    DeltaMeta(method = ..., [ inverse = ... ])

`DeltaMeta` structure specifies the approximation method for the outbound messages in the `DeltaFn` node. 

# Arguments
- `method`: required, the approximation method, currently supported methods are [`Linearization`](@ref), [`Unscented`](@ref) and [`CVI`](@ref).
- `inverse`: optional, if no inverse provided, the backward rule will be computed based on RTS (Petersen et al. 2018; On Approximate Delta Gaussian Message Passing on Factor Graphs)

Is is also possible to pass the `AbstractApproximationMethod` to the meta of the delta node directly. In this case `inverse` is set to `nothing`.

See also: [`DeltaFn`](@ref), [`Linearization`](@ref), [`Unscented`](@ref), [`CVI`](@ref).
"""
struct DeltaMeta{M, I}
    method  :: M
    inverse :: I
end

function DeltaMeta(; method::M, inverse::I = nothing) where {M, I}
    return DeltaMeta{M, I}(method, inverse)
end

getmethod(meta::DeltaMeta)          = meta.method
getinverse(meta::DeltaMeta)         = meta.inverse
getinverse(meta::DeltaMeta, k::Int) = meta.inverse[k]

import Base: map

struct DeltaFn{F} end
struct DeltaFnNode{F, P, N, S, L, M} <: AbstractFactorNode
    fn::F

    proxy::P
    out::NodeInterface
    ins::NTuple{N, IndexedNodeInterface}

    statics        :: S
    localmarginals :: L
    metadata       :: M
end

as_node_symbol(::Type{<:DeltaFn{F}}) where {F} = Symbol(replace(string(nameof(F)), "#" => ""))

functionalform(factornode::DeltaFnNode{F}) where {F}      = DeltaFn{F}
sdtype(factornode::DeltaFnNode)                           = Deterministic()
interfaces(factornode::DeltaFnNode)                       = (factornode.out, factornode.ins...)
factorisation(factornode::DeltaFnNode{F, N}) where {F, N} = ntuple(identity, N + 1)
localmarginals(factornode::DeltaFnNode)                   = factornode.localmarginals.marginals
localmarginalnames(factornode::DeltaFnNode)               = map(name, localmarginals(factornode))
metadata(factornode::DeltaFnNode)                         = factornode.metadata

collect_meta(::Type{D}, something::Nothing) where {D <: DeltaFn} = error(
    "Delta node `$(as_node_symbol(D))` requires meta specification with the `where { meta = ... }` in the `@model` macro or with the separate `@meta` specification. See documentation for the `DeltaMeta`."
)

collect_meta(::Type{<:DeltaFn}, meta::DeltaMeta) = meta
collect_meta(::Type{<:DeltaFn}, method::AbstractApproximationMethod) = DeltaMeta(; method = method, inverse = nothing)

function nodefunction(factornode::DeltaFnNode)
    # `DeltaFnNode` `nodefunction` is `δ(y - f(ins...))`
    return let f = nodefunction(factornode, Val(:out))
        (y, ins...) -> (y - f(ins...) ≈ 0) ? 1 : 0
    end
end

nodefunction(factornode::DeltaFnNode, ::Val{:out})            = factornode.proxy
nodefunction(factornode::DeltaFnNode, ::Val{:in})             = getinverse(metadata(factornode))
nodefunction(factornode::DeltaFnNode, ::Val{:in}, k::Integer) = getinverse(metadata(factornode), k)

# Rules for `::Function` objects, but with the `DeltaFn` related meta and node should redirect to the `DeltaFn` rules
function rule(::F, on, vconstraint, mnames, messages, qnames, marginals, meta::DeltaMeta, addons::Any, node::DeltaFnNode) where {F <: Function}
    return rule(DeltaFn{F}, on, vconstraint, mnames, messages, qnames, marginals, meta, addons, node)
end

function marginalrule(::F, on, mnames, messages, qnames, marginals, meta::DeltaMeta, node::DeltaFnNode) where {F <: Function}
    return marginalrule(DeltaFn{F}, on, mnames, messages, qnames, marginals, meta, node)
end

# For missing rules error msg
rule_method_error_extract_fform(::Type{<:DeltaFn}) = "DeltaFn"

# `DeltaFn` requires an access to the node function, hence, node reference is required
call_rule_is_node_required(::Type{<:DeltaFn}) = CallRuleNodeRequired()

# For `@call_rule` and `@call_marginalrule`
function call_rule_make_node(::CallRuleNodeRequired, fformtype::Type{<:DeltaFn}, nodetype::F, meta::DeltaMeta) where {F}
    # This node is not initialized properly, but we do not expect rules to access internal uninitialized fields.
    # Doing so will most likely throw an error
    return DeltaFnNode(nodetype, NodeInterface(:out, Marginalisation()), (), nothing, collect_meta(DeltaFn{F}, meta))
end

function interfaceindex(factornode::DeltaFnNode, iname::Symbol)
    # Julia's constant propagation should compile-out this if-else branch
    if iname === :out
        return 1
    elseif iname === :ins || iname === :in
        return 2
    else
        error("Unknown interface ':$(iname)' for nonlinear delta fn [ $(functionalform(factornode)) ] node")
    end
end

import FixedArguments
import FixedArguments: FixedArgument, FixedPosition

function __make_delta_fn_node(fn::F, options::FactorNodeCreationOptions, out::AbstractVariable, ins::Tuple) where {F <: Function}
    out_interface = NodeInterface(:out, Marginalisation())

    # The inputs for the deterministic function are being splitted into two groups:
    # 1. Random variables and 2. Const/Data variables (static inputs)
    randoms, statics = __split_static_inputs(ins)

    # We create interfaces only for random variables 
    # The static variables are being passed to the `FixedArguments.fix` function
    ins_interface = ntuple(i -> IndexedNodeInterface(i, NodeInterface(:in, Marginalisation())), length(randoms))

    out_index = getlastindex(out)
    connectvariable!(out_interface, out, out_index)
    setmessagein!(out, out_index, messageout(out_interface))

    foreach(zip(ins_interface, randoms)) do (in_interface, in_var)
        in_index = getlastindex(in_var)
        connectvariable!(in_interface, in_var, in_index)
        setmessagein!(in_var, in_index, messageout(in_interface))
    end

    foreach(statics) do static
        setused!(FixedArguments.value(static))
    end

    # The proxy is the actual node function, but with the static inputs already fixed at their respective position
    # We use the `__unpack_latest_static` function to get the latest value of the static variables
    proxy          = FixedArguments.fix(fn, __unpack_latest_static, statics)
    localmarginals = FactorNodeLocalMarginals((FactorNodeLocalMarginal(1, 1, :out), FactorNodeLocalMarginal(2, 2, :ins)))
    meta           = collect_meta(DeltaFn{F}, metadata(options))
    pipeline       = getpipeline(options)

    if !isnothing(pipeline)
        @warn "Delta node does not support the `pipeline` option."
    end

    return DeltaFnNode(fn, proxy, out_interface, ins_interface, statics, localmarginals, meta)
end

# This function takes the inputs of the deterministic nodes and sorts them into two 
# groups: the first group is of type `RandomVariable` and the second group is of type `ConstVariable/DataVariable`
__split_static_inputs(ins::Tuple) = __split_static_inputs(Val(1), (), (), ins)

# Stop if the `remaining` tuple is empty
__split_static_inputs(::Val{N}, randoms, statics, remaining::Tuple{}) where {N} = (randoms, statics)
# Split the `remaining` tuple into head (current) and tail (remaining)
__split_static_inputs(::Val{N}, randoms, statics, remaining::Tuple) where {N} = __split_static_inputs(Val(N), randoms, statics, first(remaining), Base.tail(remaining))

# If the current input is a random variable, we add it to the `randoms` tuple
function __split_static_inputs(::Val{N}, randoms, statics, current::RandomVariable, remaining::Tuple) where {N}
    return __split_static_inputs(Val(N + 1), (randoms..., current), statics, remaining)
end

# If the current input is a const/data variable, we add it to the `statics` tuple with its respective position
function __split_static_inputs(::Val{N}, randoms, statics, current::Union{ConstVariable, DataVariable}, remaining::Tuple) where {N}
    return __split_static_inputs(Val(N + 1), randoms, (statics..., FixedArgument(FixedPosition(N), current)), remaining)
end

# This function is used to unpack the latest value of the static variables
# For constvar we just return the value
# For datavar we get the latest value from the data stream
__unpack_latest_static(_, constvar::ConstVariable) = getconst(constvar)
__unpack_latest_static(_, datavar::DataVariable) = BayesBase.getpointmass(getdata(Rocket.getrecent(messageout(datavar, 1))))

##

function make_node(fform::F, options::FactorNodeCreationOptions, args::Vararg{<:AbstractVariable}) where {F <: Function}
    return __make_delta_fn_node(fform, options, args[1], args[2:end])
end

# By default all `meta` objects fallback to the `DeltaFnDefaultRuleLayout`
# We, however, allow for specific approximation methods to override the default `DeltaFn` rule layout for better efficiency
deltafn_rule_layout(factornode::DeltaFnNode)                  = deltafn_rule_layout(factornode, metadata(factornode))
deltafn_rule_layout(factornode::DeltaFnNode, meta::DeltaMeta) = deltafn_rule_layout(factornode, getmethod(meta), getinverse(meta))

abstract type AbstractDeltaNodeDependenciesLayout end

include("layouts/default.jl")
include("layouts/cvi.jl")

deltafn_rule_layout(::DeltaFnNode, ::AbstractApproximationMethod, inverse::Nothing)                       = DeltaFnDefaultRuleLayout()
deltafn_rule_layout(::DeltaFnNode, ::AbstractApproximationMethod, inverse::Function)                      = DeltaFnDefaultKnownInverseRuleLayout()
deltafn_rule_layout(::DeltaFnNode, ::AbstractApproximationMethod, inverse::NTuple{N, Function}) where {N} = DeltaFnDefaultKnownInverseRuleLayout()

deltafn_rule_layout(::DeltaFnNode, ::CVI, inverse::Nothing) = CVIApproximationDeltaFnRuleLayout()

function deltafn_rule_layout(::DeltaFnNode, ::CVI, inverse::Any)
    @warn "CVI Approximation does not accept the inverse function. Ignoring the provided inverse."
    return CVIApproximationDeltaFnRuleLayout()
end

function activate!(factornode::DeltaFnNode, options)
    # `DeltaFn` node may change rule arguments layout depending on the `meta`
    # This feature is similar to `functional_dependencies` for a regular `FactorNode` implementation
    return activate!(factornode, deltafn_rule_layout(factornode), options)
end

# DeltaFn is very special, so it has a very special `activate!` function
function activate!(factornode::DeltaFnNode, layout::AbstractDeltaNodeDependenciesLayout, options)
    foreach(interfaces(factornode)) do interface
        (connectedvar(interface) !== nothing) || error("Empty variable on interface $(interface) of node $(factornode)")
    end

    pipeline_stages = get_pipeline_stages(options)
    scheduler       = getscheduler(options)
    addons          = getaddons(options)

    # First we declare local marginal for `out` edge
    deltafn_apply_layout(layout, Val(:q_out), factornode, pipeline_stages, scheduler, addons)

    # Second we declare how to compute a joint marginal over all inbound edges
    deltafn_apply_layout(layout, Val(:q_ins), factornode, pipeline_stages, scheduler, addons)

    # Second we declare message passing logic for out interface
    deltafn_apply_layout(layout, Val(:m_out), factornode, pipeline_stages, scheduler, addons)

    # At last we declare message passing logic for input interfaces
    deltafn_apply_layout(layout, Val(:m_in), factornode, pipeline_stages, scheduler, addons)
end

# DeltaFn has a bit a non-standard interface layout so it has a specialised `score` function too

function score(::Type{T}, ::FactorBoundFreeEnergy, ::Deterministic, node::DeltaFnNode, skip_strategy, scheduler) where {T <: CountingReal}

    # TODO (make a function for `node.localmarginals.marginals[2]`)
    qinsmarginal = apply_skip_filter(getstream(node.localmarginals.marginals[2]), skip_strategy)

    stream  = qinsmarginal |> schedule_on(scheduler)
    mapping = (marginal) -> convert(T, -score(DifferentialEntropy(), marginal))

    return stream |> map(T, mapping)
end
