## AbstractFactorNode

# abstract type AbstractFactorNode end

# isstochastic(factornode::AbstractFactorNode)    = isstochastic(sdtype(factornode))
# isdeterministic(factornode::AbstractFactorNode) = isdeterministic(sdtype(factornode))

# interfaceindices(factornode::AbstractFactorNode, iname::Symbol)                       = interfaceindices(factornode, (iname,))
# interfaceindices(factornode::AbstractFactorNode, inames::NTuple{N, Symbol}) where {N} = map(iname -> interfaceindex(factornode, iname), inames)