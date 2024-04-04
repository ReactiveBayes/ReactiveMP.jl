
# This structure is used to create an actual node from a distribution object without creating 
# Extra constant nodes for the parameters of the distribution
struct StandaloneDistributionNode{D, C} <: AbstractFactorNode
    distribution::D
    outinterface::NodeInterface
    localclusters::C
end

functionalform(factornode::StandaloneDistributionNode) = factornode.distribution
getinterfaces(factornode::StandaloneDistributionNode) = (factornode.outinterface,)
getinterface(factornode::StandaloneDistributionNode, index) = getindex(getinterfaces(factornode), index)
getinboundinterfaces(factornode::StandaloneDistributionNode) = error("`StandaloneDistributionNode` has no inbound interfaces")
getlocalclusters(factornode::StandaloneDistributionNode) = factornode.localclusters

# The main feature of this node is that is must be created from `fform::Distribution`
# In the case it must have only one interface connected to it (the outbound edge)
# The factorization must be a single tuple with a single element as well
function factornode(fform::Distribution, interfaces, factorization)
    if !isone(length(interfaces))
        error("A factor node with a distribution object can only have one output interface.")
    end
    if !(isone(length(factorization))) || !isone(length(first(factorization)))
        error("Unsupported factorization $(factorization) for a factor node with a distribution object.")
    end
    name, variable = first(interfaces)
    outinterface = NodeInterface(name, variable)
    localclusters = FactorNodeLocalClusters((outinterface,), ((1,),))
    return StandaloneDistributionNode(fform, outinterface, localclusters)
end

# The activation of this node is very simple, it just initializes the clusters and connects the outbound message
# The outbound message is fixed to the distribution provided during the creation of the node
function activate!(factornode::StandaloneDistributionNode, options::FactorNodeActivationOptions)
    initialize_clusters!(getlocalclusters(factornode), DefaultFunctionalDependencies(), factornode, options)
    vmessageout = of(Message(factornode.distribution, true, false, nothing))
    connect!(messageout(getinterface(factornode, 1)), vmessageout)
    return nothing
end

# The score function for this node is also very simple, it just calculates the KLDivergence between the marginal on the edge and the distribution
function score(::Type{T}, ::FactorBoundFreeEnergy, node::StandaloneDistributionNode, meta, skip_strategy, scheduler) where {T <: CountingReal}
    fnstream = let skip_strategy = skip_strategy, scheduler = scheduler
        (localmarginal) -> apply_skip_filter(getmarginal(localmarginal), skip_strategy) |> schedule_on(scheduler)
    end
    # `FactorBoundFreeEnergy` here is simply equal to `kldivergence` between the marginal and the outbound message
    stream = fnstream(first(getmarginals(getlocalclusters(node))))
    return stream |> map(T, (marginal) -> convert(T, score(KLDivergence(), marginal, node.distribution)))
end
