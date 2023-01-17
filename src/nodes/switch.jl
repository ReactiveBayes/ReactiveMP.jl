export Switch, SwitchNode

# Switch Functional Form
struct Switch{N} end

ReactiveMP.as_node_symbol(::Type{<:Switch}) = :Switch

# Special node
# Generic FactorNode implementation does not work with dynamic number of inputs
# We need to reimplement the following set of functions
# functionalform(factornode::FactorNode)          
# sdtype(factornode::FactorNode)                 
# interfaces(factornode::FactorNode)              
# factorisation(factornode::FactorNode)           
# localmarginals(factornode::FactorNode)          
# localmarginalnames(factornode::FactorNode)      
# metadata(factornode::FactorNode)                
# get_pipeline_stages(factornode::FactorNode)       
#
# setmarginal!(factornode::FactorNode, cname::Symbol, marginal)
# getmarginal!(factornode::FactorNode, localmarginal::FactorNodeLocalMarginal)
#
# functional_dependencies(factornode::FactorNode, iindex::Int)
# get_messages_observable(factornode, message_dependencies)
# get_marginals_observable(factornode, marginal_dependencies)
#
# score(::Type{T}, ::FactorBoundFreeEnergy, ::Stochastic, node::AbstractFactorNode, scheduler) where T
#
# Base.show

const SwitchNodeFactorisationSupport = Union{FullFactorisation}

struct SwitchNode{N, F <: SwitchNodeFactorisationSupport, M, P} <: AbstractFactorNode
    factorisation::F

    # Interfaces
    out    :: NodeInterface
    switch :: NodeInterface
    inputs :: NTuple{N, IndexedNodeInterface}

    meta     :: M
    pipeline :: P
end

functionalform(factornode::SwitchNode{N}) where {N} = Switch{N}
sdtype(factornode::SwitchNode)                      = Deterministic()
interfaces(factornode::SwitchNode)                  = (factornode.out, factornode.switch, factornode.inputs...)
factorisation(factornode::SwitchNode)               = factornode.factorisation
localmarginals(factornode::SwitchNode)              = error("localmarginals() function is not implemented for SwitchNode")
localmarginalnames(factornode::SwitchNode)          = error("localmarginalnames() function is not implemented for SwitchNode")
metadata(factornode::SwitchNode)                    = factornode.meta
getpipeline(factornode::SwitchNode)                 = factornode.pipeline

# link interfaces to indices
interface_get_index(::Type{Val{:Switch}}, ::Type{Val{:out}}) = 1
interface_get_index(::Type{Val{:Switch}}, ::Type{Val{:switch}}) = 2
interface_get_index(::Type{Val{:Switch}}, ::Type{Val{:inputs}}) = 3

setmarginal!(factornode::SwitchNode, cname::Symbol, marginal)                = error("setmarginal() function is not implemented for SwitchNode")
getmarginal!(factornode::SwitchNode, localmarginal::FactorNodeLocalMarginal) = error("getmarginal() function is not implemented for SwitchNode")

struct SwitchNodeFunctionalDependencies <: AbstractNodeFunctionalDependenciesPipeline end

default_functional_dependencies_pipeline(::Type{<:Switch}) = SwitchNodeFunctionalDependencies()

function functional_dependencies(::SwitchNodeFunctionalDependencies, factornode::SwitchNode{N, F}, iindex::Int) where {N, F <: FullFactorisation}
    message_dependencies = if iindex === 1
        # output depends on:
        (factornode.switch, factornode.inputs)
    elseif iindex === 2
        # switch depends on:
        (factornode.out, factornode.inputs)
    elseif 2 < iindex <= N + 2
        # k'th input depends on:
        (factornode.out, factornode.switch)
    else
        error("Bad index in functional_dependencies for SwitchNode")
    end

    marginal_dependencies = ()

    return message_dependencies, marginal_dependencies
end

# function for using hard switching
function functional_dependencies(::RequireMarginalFunctionalDependencies, factornode::SwitchNode{N, F}, iindex::Int) where {N, F <: FullFactorisation}
    message_dependencies = if iindex === 1
        # output depends on:
        (factornode.inputs,)
    elseif iindex === 2
        # switch depends on:
        (factornode.out, factornode.inputs)
    elseif 2 < iindex <= N + 2
        # k'th input depends on:
        (factornode.out,)
    else
        error("Bad index in functional_dependencies for SwitchNode")
    end

    marginal_dependencies = if iindex === 1
        # output depends on:
        (factornode.switch,)
    elseif iindex == 2
        #  switch depends on
        ()
    elseif 2 < iindex <= N + 2
        # k'th input depends on:
        (factornode.switch,)
    else
        error("Bad index in function_dependencies for SwitchNode")
    end
    # println(marginal_dependencies)
    return message_dependencies, marginal_dependencies
end

# create message observable for output or switch edge without pipeline constraints (the message towards the inputs are fine by default behaviour, i.e. they depend only on switch and output and no longer on all other inputs)
function get_messages_observable(
    factornode::SwitchNode{N, F, Nothing, FactorNodePipeline{P, EmptyPipelineStage}}, messages::Tuple{NodeInterface, NTuple{N, IndexedNodeInterface}}
) where {N, F <: FullFactorisation, P <: SwitchNodeFunctionalDependencies}
    output_or_switch_interface = messages[1]
    inputsinterfaces = messages[2]

    msgs_names = Val{(name(output_or_switch_interface), name(inputsinterfaces[1]))}
    msgs_observable =
        combineLatest((messagein(output_or_switch_interface), combineLatest(map((input) -> messagein(input), inputsinterfaces), PushNew())), PushNew()) |>
        map_to((messagein(output_or_switch_interface), ManyOf(map((input) -> messagein(input), inputsinterfaces))))
    return msgs_names, msgs_observable
end

# create an observable that is used to compute the switch with pipeline constraints
function get_messages_observable(
    factornode::SwitchNode{N, F, Nothing, FactorNodePipeline{P, EmptyPipelineStage}}, messages::Tuple{NodeInterface, NTuple{N, IndexedNodeInterface}}
) where {N, F <: FullFactorisation, P <: RequireMarginalFunctionalDependencies}
    switchinterface  = messages[1]
    inputsinterfaces = messages[2]

    msgs_names = Val{(name(switchinterface), name(inputsinterfaces[1]))}
    msgs_observable =
        combineLatest((messagein(switchinterface), combineLatest(map((input) -> messagein(input), inputsinterfaces), PushNew())), PushNew()) |>
        map_to((messagein(switchinterface), ManyOf(map((input) -> messagein(input), inputsinterfaces))))
    return msgs_names, msgs_observable
end

# create an observable that is used to compute the output with pipeline constraints
function get_messages_observable(
    factornode::SwitchNode{N, F, Nothing, FactorNodePipeline{P, EmptyPipelineStage}}, messages::Tuple{NTuple{N, IndexedNodeInterface}}
) where {N, F <: FullFactorisation, P <: RequireMarginalFunctionalDependencies}
    inputsinterfaces = messages[1]

    msgs_names = Val{(name(inputsinterfaces[1]),)}
    msgs_observable = combineLatest(map((input) -> messagein(input), inputsinterfaces), PushNew()) |> map_to((ManyOf(map((input) -> messagein(input), inputsinterfaces)),))
    return msgs_names, msgs_observable
end

# create an observable that is used to compute the input with pipeline constraints
function get_messages_observable(
    factornode::SwitchNode{N, F, Nothing, FactorNodePipeline{P, EmptyPipelineStage}}, messages::Tuple{NodeInterface}
) where {N, F <: FullFactorisation, P <: RequireMarginalFunctionalDependencies}
    outputinterface = messages[1]

    msgs_names = Val{(name(outputinterface),)}
    msgs_observable = combineLatestUpdates((messagein(outputinterface),), PushNew())
    return msgs_names, msgs_observable
end

function get_marginals_observable(
    factornode::SwitchNode{N, F, Nothing, FactorNodePipeline{P, EmptyPipelineStage}}, marginals::Tuple{NodeInterface}
) where {N, F <: FullFactorisation, P <: RequireMarginalFunctionalDependencies}
    switchinterface = marginals[1]

    marginal_names       = Val{(name(switchinterface),)}
    marginals_observable = combineLatestUpdates((getmarginal(connectedvar(switchinterface), IncludeAll()),), PushNew())

    return marginal_names, marginals_observable
end

function get_marginals_observable(factornode::SwitchNode{N, F}, marginal_dependencies::Tuple{}) where {N, F <: MeanField}
    return nothing, of(nothing)
end

as_node_functional_form(::Type{<:Switch}) = ValidNodeFunctionalForm()

# Node creation related functions

sdtype(::Type{<:Switch}) = Deterministic()

collect_factorisation(::Type{<:Switch{N}}, factorisation::FullFactorisation) where {N} = factorisation
collect_factorisation(::Type{<:Switch{N}}, factorisation::Any) where {N} = __switch_incompatible_factorisation_error()

function collect_factorisation(::Type{<:Switch{N}}, factorisation::NTuple{R, Tuple{<:Integer}}) where {N, R}
    # inputs + switch + out, equivalent to FullFactorisation 
    return (R === N + 2) ? FullFactorisation() : __switch_incompatible_factorisation_error()
end

__switch_incompatible_factorisation_error() =
    error("`SwitchNode` supports only following global factorisations: [ $(SwitchNodeFactorisationSupport) ] or manually set to equivalent via constraints")

function ReactiveMP.make_node(::Type{<:Switch{N}}, factorisation::F = FullFactorisation(), meta::M = nothing, pipeline::P = nothing) where {N, F, M, P}
    @assert typeof(factorisation) <: SwitchNodeFactorisationSupport "`SwitchNode` supports only following factorisations: [ $(SwitchNodeFactorisationSupport) ]"
    out    = NodeInterface(:out, Marginalisation())
    switch = NodeInterface(:switch, Marginalisation())
    inputs = ntuple((index) -> IndexedNodeInterface(index, NodeInterface(:inputs, Marginalisation())), N)
    return SwitchNode{N, F, M, P}(factorisation, out, switch, inputs, meta, pipeline)
end

function ReactiveMP.make_node(::Type{<:Switch}, options::FactorNodeCreationOptions, out::AbstractVariable, switch::AbstractVariable, inputs::NTuple{N, AbstractVariable}) where {N}
    node = make_node(
        Switch{N}, collect_factorisation(Switch{N}, factorisation(options)), collect_meta(Switch{N}, metadata(options)), collect_pipeline(Switch{N}, getpipeline(options))
    )

    # out
    out_index = getlastindex(out)
    connectvariable!(node.out, out, out_index)
    setmessagein!(out, out_index, messageout(node.out))

    # switch
    switch_index = getlastindex(switch)
    connectvariable!(node.switch, switch, switch_index)
    setmessagein!(switch, switch_index, messageout(node.switch))

    # inputs
    foreach(zip(node.inputs, inputs)) do (ininterface, invar)
        input_index = getlastindex(invar)
        connectvariable!(ininterface, invar, input_index)
        setmessagein!(invar, input_index, messageout(ininterface))
    end

    return node
end
