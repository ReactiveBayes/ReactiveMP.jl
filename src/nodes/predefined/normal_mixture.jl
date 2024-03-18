export NormalMixture, NormalMixtureNode
export GaussianMixture, GaussianMixtureNode

# Normal Mixture Functional Form
struct NormalMixture{N} end

ReactiveMP.as_node_symbol(::Type{<:NormalMixture}) = :NormalMixture

interfaces(::Type{<:NormalMixture}) = Val((:out, :switch, :m, :p))
alias_interface(::Type{<:NormalMixture}, ::Int64, name::Symbol) = name
as_node_functional_form(::Type{<:NormalMixture}) = ValidNodeFunctionalForm()
sdtype(::Type{<:NormalMixture}) = Stochastic()
collect_factorisation(::Type{<:NormalMixture}, factorization) = NormalMixtureNodeFactorisation()

struct NormalMixtureNodeFactorisation end

struct NormalMixtureNode{N} <: AbstractFactorNode
    out    :: NodeInterface
    switch :: NodeInterface
    means  :: NTuple{N, IndexedNodeInterface}
    precs  :: NTuple{N, IndexedNodeInterface}
end

const GaussianMixture     = NormalMixture
const GaussianMixtureNode = NormalMixtureNode

functionalform(factornode::NormalMixtureNode{N}) where {N} = NormalMixture{N}
getinterfaces(factornode::NormalMixtureNode) = (factornode.out, factornode.switch, factornode.means..., factornode.precs...)
sdtype(factornode::NormalMixtureNode) = Stochastic()

interfaceindices(factornode::NormalMixtureNode, iname::Symbol)                       = (interfaceindex(factornode, iname),)
interfaceindices(factornode::NormalMixtureNode, inames::NTuple{N, Symbol}) where {N} = map(iname -> interfaceindex(factornode, iname), inames)

function interfaceindex(factornode::NormalMixtureNode, iname::Symbol)
    if iname === :out
        return 1
    elseif iname === :switch
        return 2
    elseif iname === :m
        return 3
    elseif iname === :p
        return 4
    else
        error("Unknown interface ':$(iname)' for the [ $(functionalform(factornode)) ] node")
    end
end

function factornode(::Type{<:NormalMixture}, interfaces, factorization)
    outinterface = interfaces[findfirst(((name, variable),) -> name == :out, interfaces)]
    switchinterface = interfaces[findfirst(((name, variable),) -> name == :switch, interfaces)]
    meansinterfaces = filter(((name, variable),) -> name == :m, interfaces)
    precsinterfaces = filter(((name, variable),) -> name == :p, interfaces)

    if length(meansinterfaces) < 2 || length(precsinterfaces) < 2
        error("The number of means and precisions in `NormalMixture` must be at least 2. Got `$(length(meansinterfaces))` means and `$(length(precsinterfaces))` precisions.")
    elseif length(meansinterfaces) !== length(precsinterfaces)
        error("The number of means and precisions in `NormalMixture` must be the same. Got `$(length(meansinterfaces))` means and `$(length(precsinterfaces))` precisions.")
    elseif any(cluster -> length(cluster) !== 1, factorization)
        error("The factorization around `NormalMixture` must be the naive mean-field.")
    end

    N = length(meansinterfaces)

    return NormalMixtureNode(
        NodeInterface(outinterface...),
        NodeInterface(switchinterface...),
        ntuple(i -> IndexedNodeInterface(i, NodeInterface(meansinterfaces[i]...)), N),
        ntuple(i -> IndexedNodeInterface(i, NodeInterface(precsinterfaces[i]...)), N)
    )
end

struct NormalMixtureNodeFunctionalDependencies <: FunctionalDependencies end

function activate!(factornode::NormalMixtureNode, options::FactorNodeActivationOptions)
    return activate!(NormalMixtureNodeFunctionalDependencies(), factornode, options)
end

function functional_dependencies(::NormalMixtureNodeFunctionalDependencies, factornode::NormalMixtureNode{N}, interface, iindex::Int) where {N}
    message_dependencies = ()

    marginal_dependencies = if iindex === 1
        (factornode.switch, factornode.means, factornode.precs)
    elseif iindex === 2
        (factornode.out, factornode.means, factornode.precs)
    elseif 2 < iindex <= N + 2
        (factornode.out, factornode.switch, factornode.precs[iindex - 2])
    elseif N + 2 < iindex <= 2N + 2
        (factornode.out, factornode.switch, factornode.means[iindex - N - 2])
    else
        error("Bad index in functional_dependencies for NormalMixtureNode")
    end

    return message_dependencies, marginal_dependencies
end

function collect_latest_messages(::NormalMixtureNodeFunctionalDependencies, factornode::NormalMixtureNode{N}, message_dependencies::Tuple{}) where {N}
    return nothing, of(nothing)
end

function collect_latest_marginals(
    ::NormalMixtureNodeFunctionalDependencies,
    factornode::NormalMixtureNode{N},
    marginal_dependencies::Tuple{NodeInterface, NTuple{N, IndexedNodeInterface}, NTuple{N, IndexedNodeInterface}}
) where {N}
    varinterface    = marginal_dependencies[1]
    meansinterfaces = marginal_dependencies[2]
    precsinterfaces = marginal_dependencies[3]

    marginal_names = Val{(name(varinterface), name(meansinterfaces[1]), name(precsinterfaces[1]))}()
    marginals_observable =
        combineLatest(
            (
                getmarginal(getvariable(varinterface), IncludeAll()),
                combineLatest(map((prec) -> getmarginal(getvariable(prec), IncludeAll()), reverse(precsinterfaces)), PushNew()),
                combineLatest(map((mean) -> getmarginal(getvariable(mean), IncludeAll()), reverse(meansinterfaces)), PushNew())
            ),
            PushNew()
        ) |> map_to((
            getmarginal(getvariable(varinterface), IncludeAll()),
            ManyOf(map((mean) -> getmarginal(getvariable(mean), IncludeAll()), meansinterfaces)),
            ManyOf(map((prec) -> getmarginal(getvariable(prec), IncludeAll()), precsinterfaces))
        ))

    return marginal_names, marginals_observable
end

function collect_latest_marginals(
    ::NormalMixtureNodeFunctionalDependencies, factornode::NormalMixtureNode{N}, marginal_dependencies::Tuple{NodeInterface, NodeInterface, IndexedNodeInterface}
) where {N}
    outinterface    = marginal_dependencies[1]
    switchinterface = marginal_dependencies[2]
    varinterface    = marginal_dependencies[3]

    marginal_names       = Val{(name(outinterface), name(switchinterface), name(varinterface))}()
    marginals_observable = combineLatestUpdates((getmarginal(getvariable(outinterface), IncludeAll()), getmarginal(getvariable(switchinterface), IncludeAll()), getmarginal(getvariable(varinterface), IncludeAll())), PushNew())

    return marginal_names, marginals_observable
end

# FreeEnergy related functions

@average_energy NormalMixture (q_out::Any, q_switch::Any, q_m::ManyOf{N, Any}, q_p::ManyOf{N, Any}) where {N} = begin
    z_bar = probvec(q_switch)
    return mapreduce(+, 1:N; init = 0.0) do i
        return avg_energy_nm(variate_form(typeof(q_out)), q_out, q_m, q_p, z_bar, i)
    end
end

function avg_energy_nm(::Type{Univariate}, q_out, q_m, q_p, z_bar, i)
    return z_bar[i] * score(AverageEnergy(), NormalMeanPrecision, Val{(:out, :μ, :τ)}(), map((q) -> Marginal(q, false, false, nothing), (q_out, q_m[i], q_p[i])), nothing)
end

function avg_energy_nm(::Type{Multivariate}, q_out, q_m, q_p, z_bar, i)
    return z_bar[i] * score(AverageEnergy(), MvNormalMeanPrecision, Val{(:out, :μ, :Λ)}(), map((q) -> Marginal(q, false, false, nothing), (q_out, q_m[i], q_p[i])), nothing)
end

function score(::Type{T}, ::FactorBoundFreeEnergy, ::Stochastic, node::NormalMixtureNode{N}, meta, skip_strategy, scheduler) where {T <: CountingReal, N}
    stream = combineLatest(
        (
            getmarginal(getvariable(node.out), skip_strategy) |> schedule_on(scheduler),
            getmarginal(getvariable(node.switch), skip_strategy) |> schedule_on(scheduler),
            ManyOfObservable(combineLatest(map((mean) -> getmarginal(getvariable(mean), skip_strategy) |> schedule_on(scheduler), node.means), PushNew())),
            ManyOfObservable(combineLatest(map((prec) -> getmarginal(getvariable(prec), skip_strategy) |> schedule_on(scheduler), node.precs), PushNew()))
        ),
        PushNew()
    )

    mapping = let fform = functionalform(node), meta = meta
        (marginals) -> begin
            average_energy = score(AverageEnergy(), fform, Val{(:out, :switch, :m, :p)}(), marginals, meta)

            out_entropy     = score(DifferentialEntropy(), marginals[1])
            switch_entropy  = score(DifferentialEntropy(), marginals[2])
            means_entropies = mapreduce((m) -> score(DifferentialEntropy(), m), +, marginals[3])
            precs_entropies = mapreduce((m) -> score(DifferentialEntropy(), m), +, marginals[4])

            return convert(T, average_energy - (out_entropy + switch_entropy + means_entropies + precs_entropies))
        end
    end

    return stream |> map(T, mapping)
end