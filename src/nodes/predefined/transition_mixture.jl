export TransitionMixture, TransitionMixtureNode

# Transition Mixture Functional Form
struct TransitionMixture{N} end

ReactiveMP.as_node_symbol(::Type{<:TransitionMixture}) = :TransitionMixture

interfaces(::Type{<:TransitionMixture}) = Val((:out, :in, :switch, :matrices))
alias_interface(::Type{<:TransitionMixture}, ::Int64, name::Symbol) = name
is_predefined_node(::Type{<:TransitionMixture}) = PredefinedNodeFunctionalForm()
sdtype(::Type{<:TransitionMixture}) = Stochastic()
collect_factorisation(::Type{<:TransitionMixture}, factorization) = TransitionMixtureNodeFactorisation()

struct TransitionMixtureNodeFactorisation end

struct TransitionMixtureNode{N} <: AbstractFactorNode
    out::NodeInterface
    in::NodeInterface
    switch::NodeInterface
    matrices::NTuple{N, IndexedNodeInterface}
    local_clusters::FactorNodeLocalClusters
end

functionalform(factornode::TransitionMixtureNode{N}) where {N} = TransitionMixture{N}
getinterface(factornode::TransitionMixtureNode{N}, i::Int64) where {N} = getinterfaces(factornode)[i]
getinterfaces(factornode::TransitionMixtureNode) = (factornode.out, factornode.in, factornode.switch, factornode.matrices...)
sdtype(factornode::TransitionMixtureNode) = Stochastic()

interfaceindices(factornode::TransitionMixtureNode, iname::Symbol)                       = (interfaceindex(factornode, iname),)
interfaceindices(factornode::TransitionMixtureNode, inames::NTuple{N, Symbol}) where {N} = map(iname -> interfaceindex(factornode, iname), inames)

function interfaceindex(factornode::TransitionMixtureNode, iname::Symbol)
    if iname === :out
        return 1
    elseif iname === :in
        return 2
    elseif iname === :switch
        return 3
    elseif iname === :matrices
        return 4
    else
        error("Unknown interface ':$(iname)' for the [ $(functionalform(factornode)) ] node")
    end
end

function factornode(::Type{<:TransitionMixture}, interfaces, factorization)
    @show interfaces factorization
    outinterface = interfaces[findfirst(((name, variable),) -> name == :out, interfaces)]
    ininterface = interfaces[findfirst(((name, variable),) -> name == :in, interfaces)]
    switchinterface = interfaces[findfirst(((name, variable),) -> name == :switch, interfaces)]
    matricesinterface = filter(((name, variable),) -> name == :matrices, interfaces)
    noutinterface = NodeInterface(outinterface...)
    nininterface = NodeInterface(ininterface...)
    nswitchinterface = NodeInterface(switchinterface...)

    N = length(matricesinterface)
    nmatricesinterface = ntuple(i -> IndexedNodeInterface(i, NodeInterface(matricesinterface[i]...)), N)

    marginals = (FactorNodeLocalMarginal(:out_in_switch), FactorNodeLocalMarginal(:matrices))

    factornodelocalclusters = FactorNodeLocalClusters(marginals, ((1, 2, 3), (4:N...,)))

    if length(matricesinterface) < 2
        error("The number of matrices in `TransitionMixture` must be at least 2. Got `$(length(matricesinterface))` matrices instead.")
    end

    return TransitionMixtureNode(noutinterface, nininterface, nswitchinterface, nmatricesinterface, factornodelocalclusters)
end

struct TransitionMixtureNodeFunctionalDependencies <: FunctionalDependencies end

collect_functional_dependencies(::TransitionMixtureNode, ::Nothing) = TransitionMixtureNodeFunctionalDependencies()
collect_functional_dependencies(::TransitionMixtureNode, ::TransitionMixtureNodeFunctionalDependencies) = TransitionMixtureNodeFunctionalDependencies()
collect_functional_dependencies(::TransitionMixtureNode, ::Any) =
    error("The functional dependencies for TransitionMixtureNode must be either `Nothing` or `TransitionMixtureNodeFunctionalDependencies`")

function activate!(factornode::TransitionMixtureNode, options::FactorNodeActivationOptions)
    dependecies = collect_functional_dependencies(factornode, getdependecies(options))
    ReactiveMP.initialize_clusters!(factornode.local_clusters, DefaultFunctionalDependencies(), factornode, options)
    return activate!(dependecies, factornode, options)
end

function functional_dependencies(::TransitionMixtureNodeFunctionalDependencies, factornode::TransitionMixtureNode{N}, interface, iindex::Int) where {N}
    message_dependencies = if iindex === 1
        # message_cluster = filter(!=(:out), first(filter((c) -> :out ∈ c, clusters)))
        (factornode.in, factornode.switch)
    elseif iindex === 2
        (factornode.out, factornode.switch)
    elseif iindex === 3
        (factornode.out, factornode.in)
    else
        ()
    end

    marginal_dependencies = if iindex === 1 || iindex === 2 || iindex === 3
        (factornode.matrices,)
    elseif 3 < iindex <= N + 3
        (first(factornode.local_clusters.marginals),)
    else
        error("Bad index in functional_dependencies for TransitionMixtureNode")
    end

    return message_dependencies, marginal_dependencies
end

function collect_latest_messages(::TransitionMixtureNodeFunctionalDependencies, factornode::TransitionMixtureNode{N}, message_dependencies::Tuple{}) where {N}
    return nothing, of(nothing)
end

function collect_latest_messages(
    ::TransitionMixtureNodeFunctionalDependencies, factornode::TransitionMixtureNode{N}, message_dependencies::Tuple{NodeInterface, NodeInterface}
) where {N}
    firstvarinterface = message_dependencies[1]
    secondvarinterface = message_dependencies[2]

    message_names = Val{(name(firstvarinterface), name(secondvarinterface))}()
    messages_observable = combineLatestUpdates((messagein(firstvarinterface), messagein(secondvarinterface)), PushNew())

    return message_names, messages_observable
end

function collect_latest_marginals(
    ::TransitionMixtureNodeFunctionalDependencies, factornode::TransitionMixtureNode{N}, marginal_dependencies::Tuple{NTuple{N, IndexedNodeInterface}}
) where {N}
    matricesinterfaces = first(marginal_dependencies)

    marginal_names = Val{(name(matricesinterfaces[1]),)}()
    marginals_observable =
        combineLatest((combineLatest(map((mat) -> getmarginal(getvariable(mat), IncludeAll()), matricesinterfaces), PushNew()),), PushNew()) |>
        map_to((ManyOf(map((mat) -> getmarginal(getvariable(mat), IncludeAll()), matricesinterfaces)),))

    return marginal_names, marginals_observable
end

function collect_latest_marginals(
    ::TransitionMixtureNodeFunctionalDependencies, factornode::TransitionMixtureNode{N}, marginal_dependencies::Tuple{NodeInterface, NodeInterface, NodeInterface}
) where {N}
    outinterface = marginal_dependencies[1]
    ininterface = marginal_dependencies[2]
    switchinterface = marginal_dependencies[3]

    marginal_names       = Val{(name(outinterface), name(ininterface), name(switchinterface))}()
    marginals_observable = combineLatestUpdates((getmarginal(getvariable(outinterface), IncludeAll()), getmarginal(getvariable(ininterface), IncludeAll()), getmarginal(getvariable(switchinterface), IncludeAll())), PushNew())

    return marginal_names, marginals_observable
end

# FreeEnergy related functions

# @average_energy TransitionMixture(q_out_in_switch::Any, q_matrices::ManyOf{N, Any}) where {N} = begin
#     U = 0.0
#     for i in 1:N
#         U -=*mean(log, q_matrices[i])
