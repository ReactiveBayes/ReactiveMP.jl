export Mixture, MixtureNode

# Mixture Functional Form
struct Mixture{N} end

ReactiveMP.as_node_symbol(::Type{<:Mixture}) = :Mixture

interfaces(::Type{<:Mixture}) = Val((:out, :switch, :inputs))
alias_interface(::Type{<:Mixture}, ::Int64, name::Symbol) = name
is_predefined_node(::Type{<:Mixture}) = PredefinedNodeFunctionalForm()
sdtype(::Type{<:Mixture}) = Stochastic()
collect_factorisation(::Type{<:Mixture}, factorization) = MixtureNodeFactorisation()

struct MixtureNodeFactorisation end

struct MixtureNode{N} <: AbstractFactorNode
    """
        MixtureNode{N}

    A factor node that represents a mixture model with N components that under the hood performs Bayesian model comparison.

    # Interfaces
    - `:out`: The output interface representing the mixture distribution
    - `:switch`: The switch interface representing the mixture weights (e.g. Categorical distribution)
    - `:inputs`: The N component interfaces representing the mixture components

    # Example with `@model` from RxInfer.jl
    ```julia
    # Create a mixture of 2 Gaussians
    @model function mixture_model()
        # Switch variable (mixture weights)
        s ~ Categorical([0.3, 0.7]) 
        
        # Component distributions
        c1 ~ Normal(0.0, 1.0)
        c2 ~ Normal(5.0, 1.0)
        
        # Mixture node connecting components
        y ~ Mixture(s, [c1, c2])
    end
    ```

    Note: The `Mixture` node requires the `AddonLogScale` addon to be included in the addons. However, this addon is not available for most message update rules. RxInfer.jl, which uses `ReactiveMP.jl` under the hood, allows to pass addons in the [`infer`](https://reactivebayes.github.io/RxInfer.jl/stable/manuals/inference/overview/) function. Only for certain sum-product update rules these are included. For a detailed explanation on the `Mixture` node see the [Mixture node paper](https://www.mdpi.com/1099-4300/25/8/1138).
    """
    out    :: NodeInterface
    switch :: NodeInterface
    inputs :: NTuple{N, IndexedNodeInterface}
end

functionalform(factornode::MixtureNode{N}) where {N} = Mixture{N}
getinterfaces(factornode::MixtureNode) = (factornode.out, factornode.switch, factornode.inputs...)
sdtype(factornode::MixtureNode) = Stochastic()

interfaceindices(factornode::MixtureNode, iname::Symbol)                       = (interfaceindex(factornode, iname),)
interfaceindices(factornode::MixtureNode, inames::NTuple{N, Symbol}) where {N} = map(iname -> interfaceindex(factornode, iname), inames)

function interfaceindex(factornode::MixtureNode, iname::Symbol)
    if iname === :out
        return 1
    elseif iname === :switch
        return 2
    elseif iname === :inputs
        return 3
    else
        error("Unknown interface ':$(iname)' for the [ $(functionalform(factornode)) ] node")
    end
end

function factornode(::Type{<:Mixture}, interfaces, factorization)
    outinterface = interfaces[findfirst(((name, variable),) -> name == :out, interfaces)]
    switchinterface = interfaces[findfirst(((name, variable),) -> name == :switch, interfaces)]
    inputinterfaces = filter(((name, variable),) -> name == :inputs, interfaces)

    N = length(inputinterfaces)

    return MixtureNode(NodeInterface(outinterface...), NodeInterface(switchinterface...), ntuple(i -> IndexedNodeInterface(i, NodeInterface(inputinterfaces[i]...)), N))
end

struct MixtureNodeFunctionalDependencies <: FunctionalDependencies end

collect_functional_dependencies(::MixtureNode, ::Nothing) = MixtureNodeFunctionalDependencies()
collect_functional_dependencies(::MixtureNode, ::MixtureNodeFunctionalDependencies) = MixtureNodeFunctionalDependencies()
collect_functional_dependencies(::MixtureNode, ::RequireMarginalFunctionalDependencies) = RequireMarginalFunctionalDependencies()
collect_functional_dependencies(::MixtureNode, ::Any) = error(
    "The functional dependencies for MixtureNode must be either `Nothing` or `MixtureNodeFunctionalDependencies` or `RequireMarginalFunctionalDependencies`"
)

function activate!(factornode::MixtureNode, options::FactorNodeActivationOptions)
    dependecies = collect_functional_dependencies(factornode, getdependecies(options))
    return activate!(dependecies, factornode, options)
end

function functional_dependencies(::MixtureNodeFunctionalDependencies, factornode::MixtureNode{N}, interface, iindex::Int) where {N}
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
        error("Bad index in functional_dependencies for MixtureNode")
    end

    marginal_dependencies = ()

    return message_dependencies, marginal_dependencies
end

# function for using hard switching
function functional_dependencies(::RequireMarginalFunctionalDependencies, factornode::MixtureNode{N}, iindex::Int) where {N}
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
        error("Bad index in functional_dependencies for MixtureNode")
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
        error("Bad index in function_dependencies for MixtureNode")
    end
    # println(marginal_dependencies)
    return message_dependencies, marginal_dependencies
end

# create message observable for output or Mixture edge without pipeline constraints (the message towards the inputs are fine by default behaviour, i.e. they depend only on switch and output and no longer on all other inputs)
function collect_latest_messages(::MixtureNodeFunctionalDependencies, factornode::MixtureNode{N}, messages::Tuple{NodeInterface, NTuple{N, IndexedNodeInterface}}) where {N}
    output_or_switch_interface = messages[1]
    inputsinterfaces = messages[2]

    msgs_names = Val{(name(output_or_switch_interface), name(inputsinterfaces[1]))}()
    msgs_observable =
        combineLatest((messagein(output_or_switch_interface), combineLatest(map((input) -> messagein(input), inputsinterfaces), PushNew())), PushNew()) |>
        map_to((messagein(output_or_switch_interface), ManyOf(map((input) -> messagein(input), inputsinterfaces))))
    return msgs_names, msgs_observable
end

# create an observable that is used to compute the switch with pipeline constraints
function collect_latest_messages(::RequireMarginalFunctionalDependencies, factornode::MixtureNode{N}, messages::Tuple{NodeInterface, NTuple{N, IndexedNodeInterface}}) where {N}
    switchinterface  = messages[1]
    inputsinterfaces = messages[2]

    msgs_names = Val{(name(switchinterface), name(inputsinterfaces[1]))}()
    msgs_observable =
        combineLatest((messagein(switchinterface), combineLatest(map((input) -> messagein(input), inputsinterfaces), PushNew())), PushNew()) |>
        map_to((messagein(switchinterface), ManyOf(map((input) -> messagein(input), inputsinterfaces))))
    return msgs_names, msgs_observable
end

# create an observable that is used to compute the output with pipeline constraints
function collect_latest_messages(::RequireMarginalFunctionalDependencies, ::MixtureNode{N}, messages::Tuple{NTuple{N, IndexedNodeInterface}}) where {N}
    inputsinterfaces = messages[1]

    msgs_names = Val{(name(inputsinterfaces[1]),)}()
    msgs_observable = combineLatest(map((input) -> messagein(input), inputsinterfaces), PushNew()) |> map_to((ManyOf(map((input) -> messagein(input), inputsinterfaces)),))
    return msgs_names, msgs_observable
end

# create an observable that is used to compute the input with pipeline constraints
function collect_latest_messages(::RequireMarginalFunctionalDependencies, factornode::MixtureNode{N}, messages::Tuple{NodeInterface}) where {N}
    outputinterface = messages[1]

    msgs_names = Val{(name(outputinterface),)}()
    msgs_observable = combineLatestUpdates((messagein(outputinterface),), PushNew())
    return msgs_names, msgs_observable
end

function collect_latest_marginals(::MixtureNodeFunctionalDependencies, factornode::MixtureNode{N}, marginal_dependencies::Tuple{}) where {N}
    return nothing, of(nothing)
end

function collect_latest_marginals(::RequireMarginalFunctionalDependencies, factornode::MixtureNode{N}, marginals::Tuple{NodeInterface}) where {N}
    switchinterface = marginals[1]

    marginal_names       = Val{(name(switchinterface),)}()
    marginals_observable = combineLatestUpdates((getmarginal(getvariable(switchinterface), IncludeAll()),), PushNew())

    return marginal_names, marginals_observable
end
