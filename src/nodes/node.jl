export FactorNodeType, StochasticNode, DeterministicNode, InvalidNodeType
export as_node

abstract type FactorNodeType end

struct StochasticNodeType    <: FactorNodeType end
struct DeterministicNodeType <: FactorNodeType end
struct InvalidNodeType       <: FactorNodeType end

abstract type AbstractFactorNode end

abstract type AbstractStochasticNode    <: AbstractFactorNode end
abstract type AbstractDeterministicNode <: AbstractFactorNode end

as_node(::Type)                              = InvalidNodeType()
as_node(::Type{<:AbstractStochasticNode})    = StochasticNodeType()
as_node(::Type{<:AbstractDeterministicNode}) = DeterministicNodeType()

