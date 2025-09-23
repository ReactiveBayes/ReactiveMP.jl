export GammaMixture, GammaMixtureNode

# Gamma Mixture Functional Form
struct GammaMixture{N} end

ReactiveMP.as_node_symbol(::Type{<:GammaMixture}) = :GammaMixture

interfaces(::Type{<:GammaMixture}) = Val((:out, :switch, :a, :b))
alias_interface(::Type{<:GammaMixture}, ::Int64, name::Symbol) = name
is_predefined_node(::Type{<:GammaMixture}) = PredefinedNodeFunctionalForm()
sdtype(::Type{<:GammaMixture}) = Stochastic()
collect_factorisation(::Type{<:GammaMixture}, factorization) = GammaMixtureNodeFactorisation()

struct GammaMixtureNodeFactorisation end

struct GammaMixtureNode{N} <: AbstractFactorNode
    out    :: NodeInterface
    switch :: NodeInterface
    as     :: NTuple{N, IndexedNodeInterface}
    bs     :: NTuple{N, IndexedNodeInterface}
end

functionalform(factornode::GammaMixtureNode{N}) where {N} = GammaMixture{N}
getinterfaces(factornode::GammaMixtureNode) = (factornode.out, factornode.switch, factornode.as..., factornode.bs...)
sdtype(factornode::GammaMixtureNode) = Stochastic()

interfaceindices(factornode::GammaMixtureNode, iname::Symbol)                       = (interfaceindex(factornode, iname),)
interfaceindices(factornode::GammaMixtureNode, inames::NTuple{N, Symbol}) where {N} = map(iname -> interfaceindex(factornode, iname), inames)

function interfaceindex(factornode::GammaMixtureNode, iname::Symbol)
    if iname === :out
        return 1
    elseif iname === :switch
        return 2
    elseif iname === :a
        return 3
    elseif iname === :b
        return 4
    else
        error("Unknown interface ':$(iname)' for the [ $(functionalform(factornode)) ] node")
    end
end

function factornode(::Type{<:GammaMixture}, interfaces, factorization)
    outinterface = interfaces[findfirst(((name, variable),) -> name == :out, interfaces)]
    switchinterface = interfaces[findfirst(((name, variable),) -> name == :switch, interfaces)]
    asinterfaces = filter(((name, variable),) -> name == :a, interfaces)
    bsinterfaces = filter(((name, variable),) -> name == :b, interfaces)

    if length(asinterfaces) < 2 || length(bsinterfaces) < 2
        error("The number of `a` and `b` in `GammaMixture` must be at least 2. Got `$(length(asinterfaces))` and `$(length(bsinterfaces))`.")
    elseif length(asinterfaces) !== length(bsinterfaces)
        error("The number of `a` and `b` in `GammaMixture` must be the same. Got `$(length(asinterfaces))` and `$(length(bsinterfaces))`.")
    elseif any(cluster -> length(cluster) !== 1, factorization)
        error("The factorization around `GammaMixture` must be the naive mean-field.")
    end

    N = length(asinterfaces)

    return GammaMixtureNode(
        NodeInterface(outinterface...),
        NodeInterface(switchinterface...),
        ntuple(i -> IndexedNodeInterface(i, NodeInterface(asinterfaces[i]...)), N),
        ntuple(i -> IndexedNodeInterface(i, NodeInterface(bsinterfaces[i]...)), N)
    )
end

struct GammaMixtureNodeFunctionalDependencies <: FunctionalDependencies end

collect_functional_dependencies(::GammaMixtureNode, ::Nothing) = GammaMixtureNodeFunctionalDependencies()
collect_functional_dependencies(::GammaMixtureNode, ::GammaMixtureNodeFunctionalDependencies) = GammaMixtureNodeFunctionalDependencies()
collect_functional_dependencies(::GammaMixtureNode, ::Any) = error(
    "The functional dependencies for GammaMixtureNode must be either `Nothing` or `GammaMixtureNodeFunctionalDependencies`"
)

function activate!(factornode::GammaMixtureNode, options::FactorNodeActivationOptions)
    dependecies = collect_functional_dependencies(factornode, getdependecies(options))
    return activate!(dependecies, factornode, options)
end

function functional_dependencies(::GammaMixtureNodeFunctionalDependencies, factornode::GammaMixtureNode{N}, interface, iindex::Int) where {N}
    message_dependencies = ()

    marginal_dependencies = if iindex === 1
        (factornode.switch, factornode.as, factornode.bs)
    elseif iindex === 2
        (factornode.out, factornode.as, factornode.bs)
    elseif 2 < iindex <= N + 2
        (factornode.out, factornode.switch, factornode.bs[iindex - 2])
    elseif N + 2 < iindex <= 2N + 2
        (factornode.out, factornode.switch, factornode.as[iindex - N - 2])
    else
        error("Invalid index in functional_dependencies for GammaMixtureNode")
    end

    return message_dependencies, marginal_dependencies
end

function collect_latest_messages(::GammaMixtureNodeFunctionalDependencies, ::GammaMixtureNode{N}, message_dependencies::Tuple{}) where {N}
    return nothing, of(nothing)
end

function collect_latest_marginals(
    ::GammaMixtureNodeFunctionalDependencies, ::GammaMixtureNode{N}, marginal_dependencies::Tuple{NodeInterface, NTuple{N, IndexedNodeInterface}, NTuple{N, IndexedNodeInterface}}
) where {N}
    varinterface = marginal_dependencies[1]
    asinterfaces = marginal_dependencies[2]
    bsinterfaces = marginal_dependencies[3]

    marginal_names = Val{(name(varinterface), name(asinterfaces[1]), name(bsinterfaces[1]))}()
    marginals_observable =
        combineLatest(
            (
                getmarginal(getvariable(varinterface), IncludeAll()),
                combineLatest(map((rate) -> getmarginal(getvariable(rate), IncludeAll()), reverse(bsinterfaces)), PushNew()),
                combineLatest(map((shape) -> getmarginal(getvariable(shape), IncludeAll()), reverse(asinterfaces)), PushNew())
            ),
            PushNew()
        ) |> map_to((
            getmarginal(getvariable(varinterface), IncludeAll()),
            ManyOf(map((shape) -> getmarginal(getvariable(shape), IncludeAll()), asinterfaces)),
            ManyOf(map((rate) -> getmarginal(getvariable(rate), IncludeAll()), bsinterfaces))
        ))

    return marginal_names, marginals_observable
end

function collect_latest_marginals(
    ::GammaMixtureNodeFunctionalDependencies, ::GammaMixtureNode{N}, marginal_dependencies::Tuple{NodeInterface, NodeInterface, IndexedNodeInterface}
) where {N}
    outinterface    = marginal_dependencies[1]
    switchinterface = marginal_dependencies[2]
    varinterface    = marginal_dependencies[3]

    marginal_names       = Val{(name(outinterface), name(switchinterface), name(varinterface))}()
    marginals_observable = combineLatestUpdates((getmarginal(getvariable(outinterface), IncludeAll()), getmarginal(getvariable(switchinterface), IncludeAll()), getmarginal(getvariable(varinterface), IncludeAll())), PushNew())

    return marginal_names, marginals_observable
end

# FreeEnergy related functions

@average_energy GammaMixture (q_out::Any, q_switch::Any, q_a::ManyOf{N, Any}, q_b::ManyOf{N, GammaShapeRate}) where {N} = begin
    z_bar = probvec(q_switch)
    return mapreduce(+, 1:N; init = 0.0) do i
        return z_bar[i] * score(AverageEnergy(), GammaShapeRate, Val{(:out, :α, :β)}(), map((q) -> Marginal(q, false, false, nothing), (q_out, q_a[i], q_b[i])), nothing)
    end
end

function score(::Type{T}, ::FactorBoundFreeEnergy, ::Stochastic, node::GammaMixtureNode{N}, meta, skip_strategy, scheduler) where {T <: CountingReal, N}
    stream = combineLatest(
        (
            getmarginal(getvariable(node.out), skip_strategy) |> schedule_on(scheduler),
            getmarginal(getvariable(node.switch), skip_strategy) |> schedule_on(scheduler),
            ManyOfObservable(combineLatest(map((as) -> getmarginal(getvariable(as), skip_strategy) |> schedule_on(scheduler), node.as), PushNew())),
            ManyOfObservable(combineLatest(map((bs) -> getmarginal(getvariable(bs), skip_strategy) |> schedule_on(scheduler), node.bs), PushNew()))
        ),
        PushNew()
    )

    mapping = let fform = functionalform(node), meta = meta
        (marginals) -> begin
            average_energy = score(AverageEnergy(), fform, Val{(:out, :switch, :a, :b)}(), marginals, meta)

            out_entropy    = score(DifferentialEntropy(), marginals[1])
            switch_entropy = score(DifferentialEntropy(), marginals[2])
            a_entropies    = mapreduce((m) -> score(DifferentialEntropy(), m), +, marginals[3])
            b_entropies    = mapreduce((m) -> score(DifferentialEntropy(), m), +, marginals[4])

            return convert(T, average_energy - (out_entropy + switch_entropy + a_entropies + b_entropies))
        end
    end

    return stream |> map(T, mapping)
end

## Extra distribution for the Gamma Mixture

"""
    ν(x) ∝ exp(p*β*x - p*logГ(x)) ≡ exp(γ*x - p*logГ(x))
"""
struct GammaShapeLikelihood{T <: Real} <: ContinuousUnivariateDistribution
    p::T
    γ::T # p * β
end

Distributions.params(distribution::GammaShapeLikelihood) = (distribution.p, distribution.γ)

Distributions.@distr_support GammaShapeLikelihood 0.0 Inf

BayesBase.support(dist::GammaShapeLikelihood) = Distributions.RealInterval(0.0, Inf)
BayesBase.logpdf(distribution::GammaShapeLikelihood, x::Real) = distribution.γ * x - distribution.p * loggamma(x)

BayesBase.default_prod_rule(::Type{<:GammaShapeLikelihood}, ::Type{<:GammaShapeLikelihood}) = PreserveTypeProd(Distribution)

function prod(::PreserveTypeProd{Distribution}, left::GammaShapeLikelihood, right::GammaShapeLikelihood)
    return GammaShapeLikelihood(left.p + right.p, left.γ + right.γ)
end
