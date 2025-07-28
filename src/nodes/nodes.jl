export Deterministic, Stochastic, isdeterministic, isstochastic, sdtype
export Marginalisation, MomentMatching
export functionalform, getinterfaces, factorisation, localmarginals, localmarginalnames
export FactorNode, factornode
export @node

using MessagePassingRulesBase

export MessagePassingRulesBase

import MessagePassingRulesBase: PredefinedNodeFunctionalForm, UndefinedNodeFunctionalForm, is_predefined_node
import MessagePassingRulesBase: Deterministic, Stochastic, isdeterministic, isstochastic, sdtype
import MessagePassingRulesBase: nodefunction
import MessagePassingRulesBase: as_node_symbol
import MessagePassingRulesBase: collect_factorisation, collect_meta, default_meta
import MessagePassingRulesBase: AbstractFactorNode, functionalform, getinterfaces, getinterface, getinboundinterfaces, getlocalclusters, interfaceindex, interfaceindices
import MessagePassingRulesBase: interfaces, inputinterfaces, alias_interface

using TupleTools

struct Marginalisation end
struct MomentMatching end

include("interfaces.jl")
include("clusters.jl")
include("dependencies.jl")

"""
    FactorNode(functionalform, interfaces, localclusters)

Generic factor node object that represents a factor node with a given `functionalform` and `interfaces`.
"""
struct FactorNode{F, I, C} <: AbstractFactorNode
    fform::F
    interfaces::I
    localclusters::C

    FactorNode(fform::Type{F}, interfaces::I, localclusters::C) where {F, I, C} = new{Type{F}, I, C}(fform, interfaces, localclusters)
    FactorNode(fform::F, interfaces::I, localclusters::C) where {F <: Function, I, C} = new{F, I, C}(fform, interfaces, localclusters)
end

function factornode(fform::F, interfaces::I, factorization) where {F, I}
    return factornode(MessagePassingRulesBase.is_predefined_node(fform), fform, interfaces, factorization)
end

# `PredefinedNodeFunctionalForm` are generally the nodes that are defined with the `@node` macro
# The `UndefinedNodeFunctionalForm` nodes can be created as well, but only if the `fform` is a `Function` (see `predefined/delta.jl`)
function factornode(::PredefinedNodeFunctionalForm, fform::F, interfaces::I, factorization) where {F, I}
    processed_interfaces = prepare_interfaces_generic(fform, interfaces)
    localclusters = FactorNodeLocalClusters(processed_interfaces, MessagePassingRulesBase.collect_factorisation(fform, factorization))
    return FactorNode(fform, processed_interfaces, localclusters)
end

MessagePassingRulesBase.functionalform(factornode::FactorNode) = factornode.fform
MessagePassingRulesBase.getinterfaces(factornode::FactorNode) = factornode.interfaces
MessagePassingRulesBase.getinterface(factornode::FactorNode, index) = factornode.interfaces[index]
# `getinboundinterfaces` skips the first interface, which is assumed to be the output interface
MessagePassingRulesBase.getinboundinterfaces(factornode::FactorNode) = view(factornode.interfaces, (firstindex(factornode.interfaces) + 1):lastindex(factornode.interfaces))
MessagePassingRulesBase.getlocalclusters(factornode::FactorNode) = factornode.localclusters
MessagePassingRulesBase.sdtype(factornode::FactorNode) = MessagePassingRulesBase.sdtype(MessagePassingRulesBase.functionalform(factornode))

MessagePassingRulesBase.interfaceindex(factornode::FactorNode, iname::Symbol)                         = findfirst(interface -> name(interface) === iname, getinterfaces(factornode))
MessagePassingRulesBase.interfaceindices(factornode::FactorNode, iname::Symbol)                       = (interfaceindex(factornode, iname),)
MessagePassingRulesBase.interfaceindices(factornode::FactorNode, inames::NTuple{N, Symbol}) where {N} = map(iname -> interfaceindex(factornode, iname), inames)

function prepare_interfaces_generic(fform::F, interfaces::AbstractVector) where {F}
    prepare_interfaces_check_nonempty(fform, interfaces)
    prepare_interfaces_check_adjacent_duplicates(fform, interfaces)
    prepare_interfaces_check_numarguments(fform, interfaces)
    return map(enumerate(interfaces)) do (index, (name, variable))
        return NodeInterface(MessagePassingRulesBase.alias_interface(fform, index, name), variable)
    end
end

function prepare_interfaces_check_nonempty(fform, interfaces)
    length(interfaces) > 0 || error(lazy"At least one argument is required for a factor node. Got none for `$(fform)`")
end

function prepare_interfaces_check_adjacent_duplicates(fform, interfaces)
    # Here we create an iterator that checks ONLY adjacent interfaces 
    # The reason here is that we don't want to check all possible combinations of all input interfaces
    # because that would require allocating an intermediate storage for `Set`, which would harm the 
    # performance of nodes creation. The `zip(interfaces, Iterators.drop(interfaces, 1))` creates a generic 
    # iterator of adjacent interface pairs
    foreach(zip(interfaces, Iterators.drop(interfaces, 1))) do (left, right)
        lname, _ = left
        rname, _ = right
        if isequal(lname, rname)
            error(
                lazy"`$fform` has duplicate entry for interface `$lname`. Did you pass an array (e.g. `x`) instead of an array element (e.g. `x[i]`)? Check your variable indices."
            )
        end
    end
end

function prepare_interfaces_check_numarguments(fform::F, interfaces) where {F}
    prepare_interfaces_check_num_inputarguments(fform, MessagePassingRulesBase.inputinterfaces(fform), interfaces)
end

function prepare_interfaces_check_num_inputarguments(fform, inputinterfaces::Val{Input}, interfaces) where {Input}
    (length(interfaces) - 1) === length(Input) ||
        error(lazy"Expected $(length(Input)) input arguments for `$(fform)`, got $(length(interfaces) - 1): $(join(map(first, Iterators.drop(interfaces, 1)), \", \"))")
end

struct FactorNodeActivationOptions{M, D, P, A, S, R}
    metadata::M
    dependencies::D
    pipeline::P
    addons::A
    scheduler::S
    rulefallback::R
end

getmetadata(options::FactorNodeActivationOptions) = options.metadata
getdependecies(options::FactorNodeActivationOptions) = options.dependencies
getpipeline(options::FactorNodeActivationOptions) = options.pipeline
getaddons(options::FactorNodeActivationOptions) = options.addons
getscheduler(options::FactorNodeActivationOptions) = options.scheduler
getrulefallback(options::FactorNodeActivationOptions) = options.rulefallback

# Users can override the dependencies if they want to
collect_functional_dependencies(fform::F, options::FactorNodeActivationOptions) where {F} = collect_functional_dependencies(fform, getdependecies(options))

function activate!(factornode::FactorNode, options::FactorNodeActivationOptions)
    dependencies = collect_functional_dependencies(MessagePassingRulesBase.functionalform(factornode), options)
    initialize_clusters!(MessagePassingRulesBase.getlocalclusters(factornode), dependencies, factornode, options)
    return activate!(dependencies, factornode, options)
end
