export GammaMixture, GammaMixtureNode, GammaMixtureNodeMetadata

# Gamma Mixture Functional Form
struct GammaMixture{N} end

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
#
const GammaMixtureNodeFactorisationSupport = Union{MeanField, }

struct GammaMixtureNodeMetadata{A}
    shape_likelihood_approximation :: A
end

get_shape_likelihood_approximation(meta::GammaMixtureNodeMetadata) = meta.shape_likelihood_approximation

struct GammaMixtureNode{N, F <: GammaMixtureNodeFactorisationSupport, M <: GammaMixtureNodeMetadata, P} <: AbstractFactorNode
    factorisation :: F

    # Interfaces
    out    :: NodeInterface
    switch :: NodeInterface
    as  :: NTuple{N, IndexedNodeInterface}
    bs  :: NTuple{N, IndexedNodeInterface}

    meta     :: M
    pipeline :: P
end

functionalform(factornode::GammaMixtureNode{N}) where N = GammaMixture{N}
sdtype(factornode::GammaMixtureNode)                    = Stochastic()
interfaces(factornode::GammaMixtureNode)                = (factornode.out, factornode.switch, factornode.as..., factornode.bs...)
factorisation(factornode::GammaMixtureNode)             = factornode.factorisation
localmarginals(factornode::GammaMixtureNode)            = error("localmarginals() function is not implemented for GammaMixtureNode")
localmarginalnames(factornode::GammaMixtureNode)        = error("localmarginalnames() function is not implemented for GammaMixtureNode")
metadata(factornode::GammaMixtureNode)                  = factornode.meta
get_pipeline_stages(factornode::GammaMixtureNode)       = factornode.pipeline

setmarginal!(factornode::GammaMixtureNode, cname::Symbol, marginal)                = error("setmarginal() function is not implemented for GammaMixtureNode")
getmarginal!(factornode::GammaMixtureNode, localmarginal::FactorNodeLocalMarginal) = error("getmarginal() function is not implemented for GammaMixtureNode")

## Metadata

const DefaultGammaMixtureMetadata = GammaMixtureNodeMetadata(GaussLaguerreQuadrature(32))

collect_meta(fform::Type{ <: GammaMixture }, meta::Any)                      = error("Invalid meta object $(meta) passed to GammaMixture node.")
collect_meta(fform::Type{ <: GammaMixture }, meta::GammaMixtureNodeMetadata) = meta
collect_meta(fform::Type{ <: GammaMixture }, meta::Nothing)                  = DefaultGammaMixtureMetadata

## activate!

function functional_dependencies(factornode::GammaMixtureNode{N, F}, iindex::Int) where { N, F <: MeanField }
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

function get_messages_observable(factornode::GammaMixtureNode{N, F}, message_dependencies::Tuple{}) where { N, F <: MeanField }
    return nothing, of(nothing)
end

function get_marginals_observable(
    factornode::GammaMixtureNode{N, F},
    marginal_dependencies::Tuple{ NodeInterface, NTuple{N, IndexedNodeInterface}, NTuple{N, IndexedNodeInterface} }) where { N, F <: MeanField }

    varinterface = marginal_dependencies[1]
    asinterfaces = marginal_dependencies[2]
    bsinterfaces = marginal_dependencies[3]

    marginal_names = Val{ (name(varinterface), name(asinterfaces[1]), name(bsinterfaces[1])) }
    marginals_observable = combineLatest((
        getmarginal(connectedvar(varinterface), IncludeAll()),
        combineLatest(map((rate) -> getmarginal(connectedvar(rate), IncludeAll()), reverse(bsinterfaces)), PushNew()),
        combineLatest(map((shape) -> getmarginal(connectedvar(shape), IncludeAll()), reverse(asinterfaces)), PushNew()),
    ), PushNew()) |> map_to((
        getmarginal(connectedvar(varinterface), IncludeAll()),
        map((shape) -> getmarginal(connectedvar(shape), IncludeAll()), asinterfaces),
        map((rate) -> getmarginal(connectedvar(rate), IncludeAll()), bsinterfaces)
    ))

    return marginal_names, marginals_observable
end

function get_marginals_observable(
    factornode::GammaMixtureNode{N, F},
    marginal_dependencies::Tuple{ NodeInterface, NodeInterface, IndexedNodeInterface }) where { N, F <: MeanField }

    outinterface    = marginal_dependencies[1]
    switchinterface = marginal_dependencies[2]
    varinterface    = marginal_dependencies[3]

    marginal_names       = Val{ (name(outinterface), name(switchinterface), name(varinterface)) }
    marginals_observable = combineLatestUpdates((
        getmarginal(connectedvar(outinterface), IncludeAll()),
        getmarginal(connectedvar(switchinterface), IncludeAll()),
        getmarginal(connectedvar(varinterface), IncludeAll())
    ), PushNew())

    return marginal_names, marginals_observable
end

# FreeEnergy related functions

@average_energy GammaMixture (q_out::Any, q_switch::Any, q_a::NTuple{N, Any}, q_b::NTuple{N, GammaShapeRate}) where N = begin
    z_bar = probvec(q_switch)
    return mapreduce(+, 1:N, init = 0.0) do i
        return z_bar[i] * score(AverageEnergy(), GammaShapeRate, Val{ (:out, :α , :β) }, map((q) -> Marginal(q, false, false), (q_out, q_a[i], q_b[i])), nothing)
    end
end

function score(::Type{T}, objective::BetheFreeEnergy, ::FactorBoundFreeEnergy, ::Stochastic, node::GammaMixtureNode{N, MeanField}, scheduler) where { T <: InfCountingReal, N }

    skip_strategy = marginal_skip_strategy(objective)

    stream = combineLatest((
        getmarginal(connectedvar(node.out), skip_strategy) |> schedule_on(scheduler),
        getmarginal(connectedvar(node.switch), skip_strategy) |> schedule_on(scheduler),
        combineLatest(map((as) -> getmarginal(connectedvar(as), skip_strategy) |> schedule_on(scheduler), node.as), PushNew()),
        combineLatest(map((bs) -> getmarginal(connectedvar(bs), skip_strategy) |> schedule_on(scheduler), node.bs), PushNew())
    ), PushNew())

    mapping = let fform = functionalform(node), meta = metadata(node)
        (marginals) -> begin
            average_energy   = score(AverageEnergy(), fform, Val{ (:out, :switch, :a, :b) }, marginals, meta)

            out_entropy     = score(DifferentialEntropy(), marginals[1])
            switch_entropy  = score(DifferentialEntropy(), marginals[2])
            a_entropies = mapreduce((m) -> score(DifferentialEntropy(), m), +, marginals[3])
            b_entropies = mapreduce((m) -> score(DifferentialEntropy(), m), +, marginals[4])

            return convert(T, average_energy - (out_entropy + switch_entropy + a_entropies + b_entropies))
        end
    end

    return stream |> map(T, mapping)
end

as_node_functional_form(::Type{ <: GammaMixture }) = ValidNodeFunctionalForm()

# Node creation related functions

sdtype(::Type{ <: GammaMixture }) = Stochastic()

collect_factorisation(::Type{ <: GammaMixture }, factorisation) = factorisation

function ReactiveMP.make_node(::Type{ <: GammaMixture{N} }; factorisation::F = MeanField(), meta::M = nothing, pipeline::P = EmptyPipelineStage()) where { N, F, M, P }
    @assert N >= 2 "GammaMixtureNode requires at least two mixtures on input"
    @assert typeof(factorisation) <: GammaMixtureNodeFactorisationSupport "GammaMixtureNode supports only following factorisations: [ $(GammaMixtureNodeFactorisationSupport) ]"
    out    = NodeInterface(:out)
    switch = NodeInterface(:switch)
    as   = ntuple((index) -> IndexedNodeInterface(index, NodeInterface(:a)), N)
    bs   = ntuple((index) -> IndexedNodeInterface(index, NodeInterface(:b)), N)
    meta = collect_meta(GammaMixture, meta)
    return GammaMixtureNode{N, F, typeof(meta), P}(factorisation, out, switch, as, bs, meta, pipeline)
end

function ReactiveMP.make_node(::Type{ <: GammaMixture }, out::AbstractVariable, switch::AbstractVariable, as::NTuple{N, AbstractVariable}, bs::NTuple{N, AbstractVariable}; factorisation = MeanField(), meta = nothing, pipeline = EmptyPipelineStage()) where { N}
    node = make_node(GammaMixture{N}, factorisation = collect_factorisation(GammaMixture, factorisation), meta = collect_meta(GammaMixture, meta), pipeline = pipeline)

    # out
    out_index = getlastindex(out)
    connectvariable!(node.out, out, out_index)
    setmessagein!(out, out_index, messageout(node.out))

    # switch
    switch_index = getlastindex(switch)
    connectvariable!(node.switch, switch, switch_index)
    setmessagein!(switch, switch_index, messageout(node.switch))

    # as
    foreach(zip(node.as, as)) do (ainterface, avar)
        shape_index = getlastindex(avar)
        connectvariable!(ainterface, avar, shape_index)
        setmessagein!(avar, shape_index, messageout(ainterface))
    end

    # bs
    foreach(zip(node.bs, bs)) do (binterface, bvar)
        rate_index = getlastindex(bvar)
        connectvariable!(binterface, bvar, rate_index)
        setmessagein!(bvar, rate_index, messageout(binterface))
    end

    return node
end

function ReactiveMP.make_node(fform::Type{ <: GammaMixture }, autovar::AutoVar, args::Vararg{ <: AbstractVariable }; kwargs...)
    var  = randomvar(getname(autovar))
    node = make_node(fform, var, args...; kwargs...)
    return node, var
end
