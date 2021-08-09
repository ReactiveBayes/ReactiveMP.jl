export BIFM_helper, functional_dependencies

struct BIFM_helper <: AbstractFactorNode end

@node BIFM_helper Stochastic [ output, input ]

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