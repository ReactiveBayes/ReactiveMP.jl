# todo: check dimension

export DenseReLU, DenseReLUNode, DenseReLUMeta

# DenseReLU Functional Form
struct DenseReLU{N} end

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
const DenseReLUNodeFactorisationSupport = Union{MeanField, }

struct DenseReLUNode{N, F <: DenseReLUNodeFactorisationSupport, M, P} <: AbstractFactorNode
    factorisation :: F
    
    # Interfaces
    output  :: NodeInterface
    input   :: NodeInterface
    w       :: NTuple{N, IndexedNodeInterface}
    z       :: NTuple{N, IndexedNodeInterface}
    f       :: NTuple{N, IndexedNodeInterface}

    meta     :: M
    pipeline :: P
end

functionalform(factornode::DenseReLUNode{N}) where N = DenseReLU{N}
sdtype(factornode::DenseReLUNode)                    = Stochastic()           
interfaces(factornode::DenseReLUNode)                = (factornode.output, factornode.input, factornode.w..., factornode.z..., factornode.f...)
factorisation(factornode::DenseReLUNode)             = factornode.factorisation       
localmarginals(factornode::DenseReLUNode)            = error("localmarginals() function is not implemented for DenseReLUNode")           
localmarginalnames(factornode::DenseReLUNode)        = error("localmarginalnames() function is not implemented for DenseReLUNode")     
metadata(factornode::DenseReLUNode)                  = factornode.meta            
getpipeline(factornode::DenseReLUNode)       = factornode.pipeline

setmarginal!(factornode::DenseReLUNode, cname::Symbol, marginal)                = error("setmarginal() function is not implemented for DenseReLUNode")           
getmarginal!(factornode::DenseReLUNode, localmarginal::FactorNodeLocalMarginal) = error("getmarginal() function is not implemented for DenseReLUNode")           

## activate!

struct DenseReLUNodeFunctionalDependencies <: AbstractNodeFunctionalDependenciesPipeline end 

default_functional_dependencies_pipeline(::Type{ <: DenseReLU }) = DenseReLUNodeFunctionalDependencies()

function functional_dependencies(::DenseReLUNodeFunctionalDependencies, factornode::DenseReLUNode{N, F}, iindex::Int) where { N, F <: MeanField }
    message_dependencies = ()

    marginal_dependencies = if iindex === 1 
        # output
        (factornode.input, factornode.w, factornode.z, factornode.f)
    elseif iindex === 2
        # input
        (factornode.output, factornode.w, factornode.z, factornode.f)
    elseif 2 < iindex <= N + 2
        # w
        (factornode.output, factornode.input, factornode.z[ iindex - 2 ], factornode.f[ iindex - 2])
    elseif N + 2 < iindex <= 2N + 2
        # z
        (factornode.output, factornode.input, factornode.w[ iindex - N - 2 ], factornode.f[ iindex - N - 2 ])
    elseif 2N + 2 < iindex <= 3N + 2
        # f
        (factornode.output, factornode.input, factornode.w[ iindex - 2N - 2 ], factornode.z[ iindex - 2N - 2 ])
    else
        error("Bad index in functional_dependencies for DenseReLUNode")
    end

    return message_dependencies, marginal_dependencies
end

function get_messages_observable(factornode::DenseReLUNode{N, F}, message_dependencies::Tuple{}) where { N, F <: MeanField }
    return nothing, of(nothing)
end

function get_marginals_observable(
    factornode::DenseReLUNode{N, F}, 
    marginal_dependencies::Tuple{ NodeInterface, NTuple{N, IndexedNodeInterface}, NTuple{N, IndexedNodeInterface}, NTuple{N, IndexedNodeInterface} }) where { N, F <: MeanField }

    varinterface    = marginal_dependencies[1]
    winterfaces     = marginal_dependencies[2]
    zinterfaces     = marginal_dependencies[3]
    finterfaces     = marginal_dependencies[4]

    marginal_names = Val{ (name(varinterface), name(winterfaces[1]), name(zinterfaces[1]), name(finterfaces[1])) }
    marginals_observable = combineLatest((
        getmarginal(connectedvar(varinterface), IncludeAll()),
        combineLatest(map((w) -> getmarginal(connectedvar(w), IncludeAll()), reverse(winterfaces)), PushNew()),
        combineLatest(map((z) -> getmarginal(connectedvar(z), IncludeAll()), reverse(zinterfaces)), PushNew()),
        combineLatest(map((f) -> getmarginal(connectedvar(f), IncludeAll()), reverse(finterfaces)), PushNew()),
    ), PushNew()) |> map_to((
        getmarginal(connectedvar(varinterface), IncludeAll()),
        map((w) -> getmarginal(connectedvar(w), IncludeAll()), winterfaces),
        map((z) -> getmarginal(connectedvar(z), IncludeAll()), zinterfaces),
        map((f) -> getmarginal(connectedvar(f), IncludeAll()), finterfaces)
    ))

    return marginal_names, marginals_observable
end

function get_marginals_observable(
    factornode::DenseReLUNode{N, F}, 
    marginal_dependencies::Tuple{ NodeInterface, NodeInterface, IndexedNodeInterface, IndexedNodeInterface }) where { N, F <: MeanField }

    outputinterface    = marginal_dependencies[1]
    inputinterface     = marginal_dependencies[2]
    var1interface      = marginal_dependencies[3]
    var2interface      = marginal_dependencies[4]

    marginal_names       = Val{ (name(outputinterface), name(inputinterface), name(var1interface), name(var2interface)) }
    marginals_observable = combineLatestUpdates((
        getmarginal(connectedvar(outputinterface), IncludeAll()),
        getmarginal(connectedvar(inputinterface), IncludeAll()),
        getmarginal(connectedvar(var1interface), IncludeAll()),
        getmarginal(connectedvar(var2interface), IncludeAll()),
    ), PushNew())

    return marginal_names, marginals_observable
end

# FreeEnergy related functions

# @average_energy NormalMixture (q_out::Any, q_switch::Any, q_m::NTuple{N, NormalMeanVariance}, q_p::NTuple{N, GammaDistributionsFamily}) where N = begin
#     z_bar = probvec(q_switch)
#     return mapreduce(+, 1:N, init = 0.0) do i
#         return z_bar[i] * score(AverageEnergy(), NormalMeanPrecision, Val{ (:out, :μ, :τ) }, map((q) -> Marginal(q, false, false), (q_out, q_m[i], q_p[i])), nothing)
#     end
# end

# @average_energy NormalMixture (q_out::Any, q_switch::Any, q_m::NTuple{N, MvNormalMeanCovariance}, q_p::NTuple{N, Wishart}) where N = begin
#     z_bar = probvec(q_switch)
#     return mapreduce(+, 1:N, init = 0.0) do i
#         return z_bar[i] * score(AverageEnergy(), MvNormalMeanPrecision, Val{ (:out, :μ, :Λ) }, map((q) -> Marginal(q, false, false), (q_out, q_m[i], q_p[i])), nothing)
#     end
# end

# function score(::Type{T}, objective::BetheFreeEnergy, ::FactorBoundFreeEnergy, ::Stochastic, node::NormalMixtureNode{N, MeanField}, scheduler) where { T <: InfCountingReal, N }
    
#     skip_strategy = marginal_skip_strategy(objective)

#     stream = combineLatest((
#         getmarginal(connectedvar(node.out), skip_strategy) |> schedule_on(scheduler),
#         getmarginal(connectedvar(node.switch), skip_strategy) |> schedule_on(scheduler),
#         combineLatest(map((mean) -> getmarginal(connectedvar(mean), skip_strategy) |> schedule_on(scheduler), node.means), PushNew()),
#         combineLatest(map((prec) -> getmarginal(connectedvar(prec), skip_strategy) |> schedule_on(scheduler), node.precs), PushNew())
#     ), PushNew())

#     mapping = let fform = functionalform(node), meta = metadata(node)
#         (marginals) -> begin 
#             average_energy   = score(AverageEnergy(), fform, Val{ (:out, :switch, :m, :p) }, marginals, meta)

#             out_entropy     = score(DifferentialEntropy(), marginals[1])
#             switch_entropy  = score(DifferentialEntropy(), marginals[2])
#             means_entropies = mapreduce((m) -> score(DifferentialEntropy(), m), +, marginals[3])
#             precs_entropies = mapreduce((m) -> score(DifferentialEntropy(), m), +, marginals[4])

#             return convert(T, average_energy - (out_entropy + switch_entropy + means_entropies + precs_entropies))
#         end
#     end

#     return stream |> map(T, mapping)
# end
#
# as_node_functional_form(::Type{ <: NormalMixture }) = ValidNodeFunctionalForm()

# Node creation related functions

sdtype(::Type{ <: DenseReLU }) = Stochastic()

collect_factorisation(::Type{ <: DenseReLU }, factorisation) = factorisation
        
function ReactiveMP.make_node(::Type{ <: DenseReLU{N} }; factorisation::F = MeanField(), meta::M = nothing, pipeline::P = nothing) where { N, F, M, P }
    # @assert N >= 2 "DenseReLU requires at least two mixtures on input"
    @assert typeof(factorisation) <: DenseReLUNodeFactorisationSupport "DenseReLUNode supports only following factorisations: [ $(DenseReLUNodeFactorisationSupport) ]"
    output  = NodeInterface(:output)
    input   = NodeInterface(:input)
    w       = ntuple((index) -> IndexedNodeInterface(index, NodeInterface(:w)), N)
    z       = ntuple((index) -> IndexedNodeInterface(index, NodeInterface(:z)), N)
    f       = ntuple((index) -> IndexedNodeInterface(index, NodeInterface(:f)), N)
    return DenseReLUNode{N, F, M, P}(factorisation, output, input, w, z, f, meta, pipeline)
end

function ReactiveMP.make_node(::Type{ <: DenseReLU }, output::AbstractVariable, input::AbstractVariable, w::NTuple{N, AbstractVariable}, z::NTuple{N, AbstractVariable}, f::NTuple{N, AbstractVariable}; factorisation = MeanField(), meta = nothing, pipeline = nothing) where { N}

    node = make_node(DenseReLU{N}, factorisation = collect_factorisation(DenseReLU, factorisation), meta = collect_meta(DenseReLU, meta), pipeline = collect_pipeline(DenseReLU, pipeline))

    # output
    output_index = getlastindex(output)
    connectvariable!(node.output, output, output_index)
    setmessagein!(output, output_index, messageout(node.output))

    # input
    input_index = getlastindex(input)
    connectvariable!(node.input, input, input_index)
    setmessagein!(input, input_index, messageout(node.input))

    # w
    foreach(zip(node.w, w)) do (winterface, wvar)
        w_index = getlastindex(wvar)
        connectvariable!(winterface, wvar, w_index)
        setmessagein!(wvar, w_index, messageout(winterface))
    end

    # z
    foreach(zip(node.z, z)) do (zinterface, zvar)
        z_index = getlastindex(zvar)
        connectvariable!(zinterface, zvar, z_index)
        setmessagein!(zvar, z_index, messageout(zinterface))
    end

    # f
    foreach(zip(node.f, f)) do (finterface, fvar)
        f_index = getlastindex(fvar)
        connectvariable!(finterface, fvar, f_index)
        setmessagein!(fvar, f_index, messageout(finterface))
    end

    return node
end

function ReactiveMP.make_node(fform::Type{ <: DenseReLU }, autovar::AutoVar, args::Vararg{ <: AbstractVariable }; kwargs...)
    var  = randomvar(getname(autovar))
    node = make_node(fform, var, args...; kwargs...)
    return node, var
end


mutable struct DenseReLUMeta{T}
    C :: T
    α :: T
    β :: T
    γ :: T
    ξ :: Array{T, 1}
end

function DenseReLUMeta(C::T1, α::T2, β::T3, γ::T4, N::Int64) where { T1, T2, T3, T4 }
    T = promote_type(T1, T2, T3, T4, Float64)
    ξ = ones(T, N)
    return DenseReLUMeta{T}(C, α, β, γ, ξ)
end

getC(meta::DenseReLUMeta)               = meta.C
getα(meta::DenseReLUMeta)               = meta.α
getβ(meta::DenseReLUMeta)               = meta.β
getγ(meta::DenseReLUMeta)               = meta.γ
getξ(meta::DenseReLUMeta)               = meta.ξ
getξk(meta::DenseReLUMeta, k::Int64)    = meta.ξ[k]

function setξ!(meta::DenseReLUMeta, ξ)
    meta.ξ = ξ
end

function setξk!(meta::DenseReLUMeta, k::Int64, ξ)
    meta.ξ[k] = ξ
end
