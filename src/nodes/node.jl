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

# Node parametrized by
# F - functional form, can be a distribution or whatever we can parametrize on
# A union of variable names, Union{:mean, :precision, :value}
# A factorisation which is a tuple of tuple of variables names, like ((:x1, :x2), (:x3, ))

using StaticArrays

struct Node{FunctionalForm, N}
    variables     :: SVector{N, Symbol}
    factorisation :: Vector{Vector{Int}}
end

function Node(::FunctionalForm, variables::SVector{N, Symbol}, factorisation) where { FunctionalForm, N }
    return Node{FunctionalForm, N}(deepcopy(variables), deepcopy(factorisation))
end

AdditionNode() = Node(+, SA[ :in1, :in2, :out ], [ SA[ 1, 2, 3 ] ])

functional_form(::Node{FunctionalForm}) where FunctionalForm = FunctionalForm
variables(node::Node) = node.variables
factorisation(node::Node) = node.factorisation

factors(node::Node) = map(factor -> map(i -> node.variables[i], factor), node.factorisation)

iscontain(node::Node, v) = findfirst(d -> d === v, variables(node)) !== nothing
isfactorised(node::Node, f) = findfirst(d -> d == f, factorisation(node)) !== nothing

node = AdditionNode()

@btime iscontain($node, :out)

factorisation(node)

@btime factors($node)

using BenchmarkTools

f = SA[ 1, 2, 3 ]
@btime isfactorised($node, $f)
