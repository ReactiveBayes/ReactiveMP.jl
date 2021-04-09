export NormalMixture, NormalMixtureNode
export GaussianMixture, GaussianMixtureNode

# Normal Mixture Functional Form
struct NormalMixture{N} end

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
# outbound_message_portal(factornode::FactorNode)       
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
# 
const NormalMixtureNodeFactorisationSupport = Union{MeanField, }

struct NormalMixtureNode{N, F <: NormalMixtureNodeFactorisationSupport, S, M, P} <: AbstractFactorNode
    factorisation :: F
    
    # Interfaces
    out    :: NodeInterface
    switch :: NodeInterface
    means  :: S
    precs  :: S

    meta   :: M
    portal :: P
end

const GaussianMixture     = NormalMixture
const GaussianMixtureNode = NormalMixtureNode

functionalform(factornode::NormalMixtureNode{N}) where N = NormalMixture{N}
sdtype(factornode::NormalMixtureNode)                    = Stochastic()           
interfaces(factornode::NormalMixtureNode)                = Iterators.flatten(((factornode.out, ), (factornode.switch, ), factornode.means, factornode.precs))
factorisation(factornode::NormalMixtureNode)             = factornode.factorisation       
localmarginals(factornode::NormalMixtureNode)            = error("localmarginals() function is not implemented for NormalMixtureNode")           
localmarginalnames(factornode::NormalMixtureNode)        = error("localmarginalnames() function is not implemented for NormalMixtureNode")     
metadata(factornode::NormalMixtureNode)                  = factornode.meta            
outbound_message_portal(factornode::NormalMixtureNode)   = factornode.portal   

setmarginal!(factornode::NormalMixtureNode, cname::Symbol, marginal)                = error("setmarginal() function is not implemented for NormalMixtureNode")           
getmarginal!(factornode::NormalMixtureNode, localmarginal::FactorNodeLocalMarginal) = error("getmarginal() function is not implemented for NormalMixtureNode")           

## activate!

function functional_dependencies(factornode::NormalMixtureNode{N, F}, iindex::Int) where { N, F <: MeanField }
    message_dependencies = ()

    marginal_dependencies = if iindex === 1
        (factornode.switch, factornode.means, factornode.precs)
    elseif iindex === 2
        (factornode.out, factornode.means, factornode.precs)
    elseif 2 < iindex <= N + 2
        (factornode.out, factornode.switch, factornode.precs[ iindex - 2 ])
    elseif N + 2 < iindex <= 2N + 2
        (factornode.out, factornode.switch, factornode.means[ iindex - N - 2 ])
    else
        error("Bad index in functional_dependencies for NormalMixtureNode")
    end

    return message_dependencies, marginal_dependencies
end

function get_messages_observable(factornode::NormalMixtureNode{N, F}, message_dependencies::Tuple{}) where { N, F <: MeanField }
    return nothing, of(nothing)
end

function get_marginals_observable(
    factornode::NormalMixtureNode{N, F}, 
    marginal_dependencies::Tuple{ NodeInterface, NTuple{N, IndexedNodeInterface}, NTuple{N, IndexedNodeInterface} }) where { N, F <: MeanField }

    varinterface    = marginal_dependencies[1]
    meansinterfaces = marginal_dependencies[2]
    precsinterfaces = marginal_dependencies[3]

    marginal_names = Val{ (name(varinterface), name(meansinterfaces[1]), name(precsinterfaces[1])) }
    marginals_observable = combineLatest((
        getmarginal(connectedvar(varinterface), IncludeAll()),
        combineLatest(map((prec) -> getmarginal(connectedvar(prec), IncludeAll()), reverse(precsinterfaces)), PushNew()),
        combineLatest(map((mean) -> getmarginal(connectedvar(mean), IncludeAll()), reverse(meansinterfaces)), PushNew()),
    ), PushNew()) |> map_to((
        getmarginal(connectedvar(varinterface), IncludeAll()),
        map((mean) -> getmarginal(connectedvar(mean), IncludeAll()), meansinterfaces),
        map((prec) -> getmarginal(connectedvar(prec), IncludeAll()), precsinterfaces)
    ))

    return marginal_names, marginals_observable
end

function get_marginals_observable(::NormalMixtureNode{N, F}, marginal_dependencies::Tuple{ NodeInterface, R, R }) where { N, R <: AbstractVector{IndexedNodeInterface}, F <: MeanField }

    varinterface    = marginal_dependencies[1]
    meansinterfaces = marginal_dependencies[2]
    precsinterfaces = marginal_dependencies[3]

    marginal_names = Val{ (name(varinterface), name(meansinterfaces[1]), name(precsinterfaces[1])) }
    marginals_observable = combineLatest((
        getmarginal(connectedvar(varinterface), IncludeAll()),
        collectLatest(Marginal, Nothing, map((prec) -> getmarginal(connectedvar(prec), IncludeAll()), reverse(precsinterfaces)), _ -> nothing),
        collectLatest(Marginal, Nothing, map((mean) -> getmarginal(connectedvar(mean), IncludeAll()), reverse(meansinterfaces)), _ -> nothing),
    ), PushNew()) |> map_to((
        getmarginal(connectedvar(varinterface), IncludeAll()),
        map((mean) -> getmarginal(connectedvar(mean), IncludeAll()), meansinterfaces),
        map((prec) -> getmarginal(connectedvar(prec), IncludeAll()), precsinterfaces)
    ))

    return marginal_names, marginals_observable
end

function get_marginals_observable(
    factornode::NormalMixtureNode{N, F}, 
    marginal_dependencies::Tuple{ NodeInterface, NodeInterface, IndexedNodeInterface }) where { N, F <: MeanField }

    outinterface    = marginal_dependencies[1]
    switchinterface = marginal_dependencies[2]
    varinterface    = marginal_dependencies[3]

    marginal_names       = Val{ (name(outinterface), name(switchinterface), name(varinterface)) }
    marginals_observable = combineLatestUpdates((
        getmarginal(connectedvar(outinterface), IncludeAll()),
        getmarginal(connectedvar(switchinterface), IncludeAll()),
        getmarginal(connectedvar(varinterface), IncludeAll()),
    ), PushNew())

    return marginal_names, marginals_observable
end

# FreeEnergy related functions

@average_energy NormalMixture (q_out::Any, q_switch::Any, q_m::NTuple{N, NormalMeanVariance}, q_p::NTuple{N, GammaDistributionsFamily}) where N = begin
    z_bar = probvec(q_switch)
    return mapreduce(+, 1:N, init = 0.0) do i
        return z_bar[i] * score(AverageEnergy(), NormalMeanPrecision, Val{ (:out, :μ, :τ) }, map((q) -> Marginal(q, false, false), (q_out, q_m[i], q_p[i])), nothing)
    end
end

@average_energy NormalMixture (q_out::Any, q_switch::Any, q_m::NTuple{N, MvNormalMeanCovariance}, q_p::NTuple{N, Wishart}) where N = begin
    z_bar = probvec(q_switch)
    return mapreduce(+, 1:N, init = 0.0) do i
        return z_bar[i] * score(AverageEnergy(), MvNormalMeanPrecision, Val{ (:out, :μ, :Λ) }, map((q) -> Marginal(q, false, false), (q_out, q_m[i], q_p[i])), nothing)
    end
end

@average_energy NormalMixture (q_out::Any, q_switch::Any, q_m::AbstractVector, q_p::AbstractVector) = begin
    @assert all(d -> variate_form(d) === variate_form(first(q_m)), q_m)
    @assert all(d -> variate_form(d) === variate_form(first(q_p)), q_p)
    @assert length(probvec(q_switch)) === length(q_m) === length(q_p)
    T = promote_variate_type(variate_form(first(q_m)), NormalMeanPrecision)
    return mapreduce(+, zip(probvec(q_switch), q_m, q_p), init = 0.0) do (z, m, p)
        return z * score(AverageEnergy(), T, Val{ node_interfaces_names(T) }, map((q) -> Marginal(q, false, false), (q_out, m, p)), nothing)
    end
end

__normal_mixture_collect_interfaces_score(marginals::NTuple)         = combineLatest(marginals, PushNew())
__normal_mixture_collect_interfaces_score(marginals::AbstractVector) = collectLatest(Marginal, AbstractVector{Marginal}, marginals, identity)

function score(::Type{T}, objective::BetheFreeEnergy, ::FactorBoundFreeEnergy, ::Stochastic, node::NormalMixtureNode{N, MeanField}, scheduler) where { T <: InfCountingReal, N }
    
    skip_strategy = marginal_skip_strategy(objective)

    stream = combineLatest((
        getmarginal(connectedvar(node.out), skip_strategy) |> schedule_on(scheduler),
        getmarginal(connectedvar(node.switch), skip_strategy) |> schedule_on(scheduler),
        __normal_mixture_collect_interfaces_score(map((mean) -> getmarginal(connectedvar(mean), skip_strategy) |> schedule_on(scheduler), node.means)),
        __normal_mixture_collect_interfaces_score(map((prec) -> getmarginal(connectedvar(prec), skip_strategy) |> schedule_on(scheduler), node.precs))
    ), PushNew())

    mapping = let fform = functionalform(node), meta = metadata(node)
        (marginals) -> begin 
            average_energy   = score(AverageEnergy(), fform, Val{ (:out, :switch, :m, :p) }, marginals, meta)

            out_entropy     = score(DifferentialEntropy(), marginals[1])
            switch_entropy  = score(DifferentialEntropy(), marginals[2])
            means_entropies = mapreduce((m) -> score(DifferentialEntropy(), m), +, marginals[3])
            precs_entropies = mapreduce((m) -> score(DifferentialEntropy(), m), +, marginals[4])

            return convert(T, average_energy - (out_entropy + switch_entropy + means_entropies + precs_entropies))
        end
    end

    return stream |> map(T, mapping)
end

as_node_functional_form(::Type{ <: NormalMixture }) = ValidNodeFunctionalForm()

# Node creation related functions

sdtype(::Type{ <: NormalMixture }) = Stochastic()

collect_factorisation(::Type{ <: NormalMixture }, factorisation) = factorisation

function make_interfaces(::Type{ <: NormalMixture }, meanvars::NTuple{N, <: AbstractVariable}, precvars::NTuple{N, <: AbstractVariable}) where N
    out    = NodeInterface(:out)
    switch = NodeInterface(:switch)
    means  = ntuple((index) -> IndexedNodeInterface(index, NodeInterface(:m)), N)
    precs  = ntuple((index) -> IndexedNodeInterface(index, NodeInterface(:p)), N)
    return out, switch, means, precs
end

function make_interfaces(::Type{ <: NormalMixture }, meanvars::Vector{ <: AbstractVariable }, precvars::Vector{ <: AbstractVariable })
    @assert length(meanvars) === length(precvars)
    N      = length(meanvars)
    out    = NodeInterface(:out)
    switch = NodeInterface(:switch)
    means  = map((index) -> IndexedNodeInterface(index, NodeInterface(:m)), 1:N)
    precs  = map((index) -> IndexedNodeInterface(index, NodeInterface(:p)), 1:N)
    return out, switch, means, precs
end

# NormalMixture node may work in two different regimes depending of the input type of the mean and precision random variables storage
# 1. First regime is a tuple based storage, which works faster for small number of mixtures, and compiles very long for large number of mixtures
# 2. Second regime is a vector based storage, which works slower than tuple in general, but compilation time is faster
# See also: `make_interfaces` function
function ReactiveMP.make_node(::Type{ <: NormalMixture }, outvar::AbstractVariable, switchvar::AbstractVariable, meanvars::R, precvars::R; factorisation = MeanField(), meta = nothing, portal = EmptyPortal()) where { R }
    @assert length(meanvars) === length(precvars) "Means and precisions inputs should have the same length"

    N = length(meanvars)

    @assert N >= 2 "NormalMixtureNode requires at least two mixtures on input"
    @assert typeof(factorisation) <: NormalMixtureNodeFactorisationSupport "NormalMixtureNode supports only following factorisations: [ $(NormalMixtureNodeFactorisationSupport) ]"

    out, switch, means, precs = make_interfaces(NormalMixture, meanvars, precvars)
    
    factorisation = collect_factorisation(NormalMixture, factorisation)
    meta          = collect_meta(NormalMixture, meta)

    F = typeof(factorisation)
    S = typeof(means)
    M = typeof(meta)
    P = typeof(portal)

    node = NormalMixtureNode{N, F, S, M, P}(factorisation, out, switch, means, precs, meta, portal)

    # out
    out_index = getlastindex(outvar)
    connectvariable!(node.out, outvar, out_index)
    setmessagein!(outvar, out_index, messageout(node.out))

    # switch
    switch_index = getlastindex(switchvar)
    connectvariable!(node.switch, switchvar, switch_index)
    setmessagein!(switchvar, switch_index, messageout(node.switch))

    # meanvars
    foreach(zip(node.means, meanvars)) do (minterface, mvar)
        mean_index = getlastindex(mvar)
        connectvariable!(minterface, mvar, mean_index)
        setmessagein!(mvar, mean_index, messageout(minterface))
    end

    # precs
    foreach(zip(node.precs, precvars)) do (pinterface, pvar)
        prec_index = getlastindex(pvar)
        connectvariable!(pinterface, pvar, prec_index)
        setmessagein!(pvar, prec_index, messageout(pinterface))
    end

    return node
end

function ReactiveMP.make_node(fform::Type{ <: NormalMixture }, autovar::AutoVar, args::Vararg{ <: AbstractVariable }; kwargs...)
    var  = randomvar(getname(autovar))
    node = make_node(fform, var, args...; kwargs...)
    return node, var
end