export BIFM_helper, functional_dependencies

@doc raw"""
The BIFM\_helper node is a node required to perform efficient message passing inconjuction with
the BIFM node. It is required to switch from the backward pass with messages to the forward pass
with marginals.

```julia
out ~ BIFM_helper(in)
```

Interfaces:
1. out - output of the BIFM_helper node, should be connected to the state space model.
2. in - input of the BIFM_helper node, should be connected to the prior for the latent state.
"""
struct BIFM_helper <: AbstractFactorNode end

@node BIFM_helper Stochastic [ out, in ]

# specify custom functional dependencies for BIFM_helper node
function functional_dependencies(factornode::FactorNode{ <: Type{ReactiveMP.BIFM_helper} }, iindex::Int)
    cindex             = clusterindex(factornode, iindex)

    nodeinterfaces     = interfaces(factornode)
    nodeclusters       = factorisation(factornode)
    nodelocalmarginals = localmarginals(factornode)

    varcluster = @inbounds nodeclusters[ cindex ]
    
    # output
    if iindex === 2
        mdependencies = (nodeinterfaces[1], )
        return tuple(mdependencies...), ()
    elseif iindex === 1
        qdependencies = TupleTools.deleteat(nodelocalmarginals, cindex)
        return (), tuple(qdependencies...) 
    end
    
end