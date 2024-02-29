export NormalMixture, NormalMixtureNode
export GaussianMixture, GaussianMixtureNode

# Normal Mixture Functional Form
struct NormalMixture{N} end

ReactiveMP.as_node_symbol(::Type{<:NormalMixture}) = :NormalMixture

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

const NormalMixtureNodeFactorisationSupport = Union{MeanField}

struct NormalMixtureNode{N, F <: NormalMixtureNodeFactorisationSupport, M, P} <: AbstractFactorNode
    factorisation::F

    # Interfaces
    out    :: NodeInterface
    switch :: NodeInterface
    means  :: NTuple{N, IndexedNodeInterface}
    precs  :: NTuple{N, IndexedNodeInterface}

    meta     :: M
    pipeline :: P
end

const GaussianMixture     = NormalMixture
const GaussianMixtureNode = NormalMixtureNode

functionalform(factornode::NormalMixtureNode{N}) where {N} = NormalMixture{N}
sdtype(factornode::NormalMixtureNode)                      = Stochastic()
interfaces(factornode::NormalMixtureNode)                  = (factornode.out, factornode.switch, factornode.means..., factornode.precs...)
factorisation(factornode::NormalMixtureNode)               = factornode.factorisation
localmarginals(factornode::NormalMixtureNode)              = error("localmarginals() function is not implemented for NormalMixtureNode")
localmarginalnames(factornode::NormalMixtureNode)          = error("localmarginalnames() function is not implemented for NormalMixtureNode")
metadata(factornode::NormalMixtureNode)                    = factornode.meta
getpipeline(factornode::NormalMixtureNode)                 = factornode.pipeline

setmarginal!(factornode::NormalMixtureNode, cname::Symbol, marginal)                = error("setmarginal() function is not implemented for NormalMixtureNode")
getmarginal!(factornode::NormalMixtureNode, localmarginal::FactorNodeLocalMarginal) = error("getmarginal() function is not implemented for NormalMixtureNode")

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

struct NormalMixtureNodeFunctionalDependencies <: AbstractNodeFunctionalDependenciesPipeline end

default_functional_dependencies_pipeline(::Type{<:NormalMixture}) = NormalMixtureNodeFunctionalDependencies()

function functional_dependencies(::NormalMixtureNodeFunctionalDependencies, factornode::NormalMixtureNode{N, F}, iindex::Int) where {N, F <: MeanField}
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

function get_messages_observable(factornode::NormalMixtureNode{N, F}, message_dependencies::Tuple{}) where {N, F <: MeanField}
    return nothing, of(nothing)
end

function get_marginals_observable(
    factornode::NormalMixtureNode{N, F}, marginal_dependencies::Tuple{NodeInterface, NTuple{N, IndexedNodeInterface}, NTuple{N, IndexedNodeInterface}}
) where {N, F <: MeanField}
    varinterface    = marginal_dependencies[1]
    meansinterfaces = marginal_dependencies[2]
    precsinterfaces = marginal_dependencies[3]

    marginal_names = Val{(name(varinterface), name(meansinterfaces[1]), name(precsinterfaces[1]))}()
    marginals_observable =
        combineLatest(
            (
                getmarginal(connected_properties(varinterface), IncludeAll()),
                combineLatest(map((prec) -> getmarginal(connected_properties(prec), IncludeAll()), reverse(precsinterfaces)), PushNew()),
                combineLatest(map((mean) -> getmarginal(connected_properties(mean), IncludeAll()), reverse(meansinterfaces)), PushNew())
            ),
            PushNew()
        ) |> map_to((
            getmarginal(connected_properties(varinterface), IncludeAll()),
            ManyOf(map((mean) -> getmarginal(connected_properties(mean), IncludeAll()), meansinterfaces)),
            ManyOf(map((prec) -> getmarginal(connected_properties(prec), IncludeAll()), precsinterfaces))
        ))

    return marginal_names, marginals_observable
end

function get_marginals_observable(factornode::NormalMixtureNode{N, F}, marginal_dependencies::Tuple{NodeInterface, NodeInterface, IndexedNodeInterface}) where {N, F <: MeanField}
    outinterface    = marginal_dependencies[1]
    switchinterface = marginal_dependencies[2]
    varinterface    = marginal_dependencies[3]

    marginal_names       = Val{(name(outinterface), name(switchinterface), name(varinterface))}()
    marginals_observable = combineLatestUpdates((getmarginal(connected_properties(outinterface), IncludeAll()), getmarginal(connected_properties(switchinterface), IncludeAll()), getmarginal(connected_properties(varinterface), IncludeAll())), PushNew())

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

function score(::Type{T}, ::FactorBoundFreeEnergy, ::Stochastic, node::NormalMixtureNode{N, MeanField}, skip_strategy, scheduler) where {T <: CountingReal, N}
    stream = combineLatest(
        (
            getmarginal(connected_properties(node.out), skip_strategy) |> schedule_on(scheduler),
            getmarginal(connected_properties(node.switch), skip_strategy) |> schedule_on(scheduler),
            ManyOfObservable(combineLatest(map((mean) -> getmarginal(connected_properties(mean), skip_strategy) |> schedule_on(scheduler), node.means), PushNew())),
            ManyOfObservable(combineLatest(map((prec) -> getmarginal(connected_properties(prec), skip_strategy) |> schedule_on(scheduler), node.precs), PushNew()))
        ),
        PushNew()
    )

    mapping = let fform = functionalform(node), meta = metadata(node)
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

as_node_functional_form(::Type{<:NormalMixture}) = ValidNodeFunctionalForm()

# Node creation related functions

sdtype(::Type{<:NormalMixture}) = Stochastic()

collect_factorisation(::Type{<:NormalMixture}, ::Nothing)                = MeanField()
collect_factorisation(::Type{<:NormalMixture}, factorisation::MeanField) = factorisation
collect_factorisation(::Type{<:NormalMixture}, factorisation::Any)       = __normal_mixture_incompatible_factorisation_error()

function collect_factorisation(::Type{<:NormalMixture{N}}, factorisation::NTuple{R, Tuple{<:Integer}}) where {N, R}
    # 2 * (m, w) + s + out, equivalent to MeanField 
    return (R === 2 * N + 2) ? MeanField() : __normal_mixture_incompatible_factorisation_error()
end

__normal_mixture_incompatible_factorisation_error() =
    error("`NormalMixtureNode` supports only following global factorisations: [ $(NormalMixtureNodeFactorisationSupport) ] or manually set to equivalent via constraints")

function ReactiveMP.make_node(::Type{<:NormalMixture{N}}, options::FactorNodeCreationOptions) where {N}
    @assert N >= 2 "`NormalMixtureNode` requires at least two mixtures on input"
    out    = NodeInterface(:out, Marginalisation())
    switch = NodeInterface(:switch, Marginalisation())
    means  = ntuple((index) -> IndexedNodeInterface(index, NodeInterface(:m, Marginalisation())), N)
    precs  = ntuple((index) -> IndexedNodeInterface(index, NodeInterface(:p, Marginalisation())), N)

    _factorisation = collect_factorisation(NormalMixture{N}, factorisation(options))
    _meta = collect_meta(NormalMixture{N}, metadata(options))
    _pipeline = collect_pipeline(NormalMixture{N}, getpipeline(options))

    @assert typeof(_factorisation) <: NormalMixtureNodeFactorisationSupport "`NormalMixtureNode` supports only following factorisations: [ $(NormalMixtureNodeFactorisationSupport) ]"

    F = typeof(_factorisation)
    M = typeof(_meta)
    P = typeof(_pipeline)

    return NormalMixtureNode{N, F, M, P}(_factorisation, out, switch, means, precs, _meta, _pipeline)
end

function ReactiveMP.make_node(
    ::Type{<:NormalMixture},
    options::FactorNodeCreationOptions,
    out::AbstractVariable,
    switch::AbstractVariable,
    means::NTuple{N, AbstractVariable},
    precs::NTuple{N, AbstractVariable}
) where {N}
    node = make_node(NormalMixture{N}, options)

    # out
    out_index = getlastindex(out)
    connectvariable!(node.out, out, out_index)
    setmessagein!(out, out_index, messageout(node.out))

    # switch
    switch_index = getlastindex(switch)
    connectvariable!(node.switch, switch, switch_index)
    setmessagein!(switch, switch_index, messageout(node.switch))

    # means
    foreach(zip(node.means, means)) do (minterface, mvar)
        mean_index = getlastindex(mvar)
        connectvariable!(minterface, mvar, mean_index)
        setmessagein!(mvar, mean_index, messageout(minterface))
    end

    # precs
    foreach(zip(node.precs, precs)) do (pinterface, pvar)
        prec_index = getlastindex(pvar)
        connectvariable!(pinterface, pvar, prec_index)
        setmessagein!(pvar, prec_index, messageout(pinterface))
    end

    return node
end
