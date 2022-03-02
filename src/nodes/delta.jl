
struct DeltaFn end

struct DeltaFnNode{F, N, L, M} <: AbstractFactorNode
    fn :: F
    
    out :: NodeInterface
    ins :: NTuple{N, IndexedNodeInterface}
    
    localmarginals :: L
    metadata       :: M
end

function __make_delta_fn_node(fn::F, out::AbstractVariable, ins::NTuple{N, <: AbstractVariable}; factorisation = nothing, meta::M = nothing) where { F <: Function, N, M }
    out_interface = NodeInterface(out, Marginalisation())
    ins_interface = ntuple(i -> IndexedNodeInterface(i, NodeInterface(:in, Marginalisation())), N)

    

    metadata = collect_meta(DeltaFn, meta)
end

function make_node(fform::F, var::AutoVar, args::Vararg{ <: AbstractVariable }; kwargs...) where { F <: Function }
    error("Unknown functional form '$(fform)' used for node specification.")
end

# make_node(fform, ::AutoVar, ::Vararg{ <: AbstractVariable }; kwargs...) = error("Unknown functional form '$(fform)' used for node specification.")
# make_node(fform, args::Vararg{ <: AbstractVariable }; kwargs...)        = error("Unknown functional form '$(fform)' used for node specification.")