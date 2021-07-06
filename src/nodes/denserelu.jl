# todo: check dimension

export DenseReLU, DenseReLUNode, DenseReLUMeta

# DenseReLU Functional Form
@doc raw"""
## General information
The `DenseReLU` function creates the `DenseReLU` factor node. For options regarding this node, please have a look at the `DenseReLUMeta` constructor function.

> **Important**: This function can only be used inside of the `@model` macro from `GraphPPL.jl` when defining a probabilistic model. The corresponding function calls and styling applies.

## Function arguments
The `DenseReLU(x, w, z, f)` function requires the following input argument:
- `x` [Required] - Random variable of dimension ``M`` specifying the input of the `DenseReLU` node.
- `w` [Required] - Tuple of random variables specifying the weights of the `DenseReLU` node. The tuple should have length ``N``, corresponding to the dimension of the output, and each element inside the tuple should have dimension ``M``.
- `z` [Required] - Tuple of random variables specifying the probability of the corresponding variable `f` being larger than 0. The tuple should be of length ``N``.
- `f` [Required] - Tuple of random variables specifying the "unfolded" multiplication of variables `x` and `W`. The tuple should be of length ``N``.

## Return values
The `DenseReLU(...)` function returns the random variable:
- `y` - Random variable of dimension ``N`` specifying the output of the `DenseReLU` node.

## Examples
Inside of the `@model` macro of `GraphPPL.jl`, the `DenseReLU` factor node can be defined as:

```julia
y ~ DenseReLU(x, w, z, f) where { meta = DenseReLUMeta(3; β=100.0) }
```

Here the `DenseReLUMeta` object encodes hyperparameter settings of this node.

"""
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
getpipeline(factornode::DenseReLUNode)               = factornode.pipeline

setmarginal!(factornode::DenseReLUNode, cname::Symbol, marginal)                = error("setmarginal() function is not implemented for DenseReLUNode")           
getmarginal!(factornode::DenseReLUNode, localmarginal::FactorNodeLocalMarginal) = error("getmarginal() function is not implemented for DenseReLUNode")       


## meta data

mutable struct DenseReLUMeta{T}
    dim_out :: Int64
    C :: T
    β :: T
    γ :: T
    ξ :: Array{T, 1}
end

@doc raw"""
## General information
The `DenseReLUMeta` structure contains the meta data of the corresponding `DenseReLU` factor node.

## Function arguments
The `DenseReLUMeta(...)` constructor function requires the following input argument:
- `dim_out` [Required] - The output dimension of the node.
Furthermore the `DenseReLUMeta(...)` constructor function allows the following keyword arguments:
- `C` [default = 1.0] - The horizontal scaling of the sigmoid function. For a large value of `C` the sigmoid function approximates the Heaviside function, which is used for approximating the ReLU function.
- `β` [default = 100.0] - The process noise precision of ``f_n``. The intermediate variable ``f_n`` is modeled as ``p(f_n \mid W_n, x) = \mathcal{N}(f_n \mid W_n^\top x, \beta^{-1})``.
- `γ` [default = 100.0] - The process noise precision of ``y_n``. The output variable ``y_n`` is modeled as ``p(y_n \mid z_n, f_n) = \mathcal{N}(y_n \mid z_n f_n, \gamma^{-1})``.

## Return values
The `DenseReLUMeta(...)` constructor function returns the `DenseReLUMeta` structure, containing all above arguments passed in the constructor function call. 
Furthermore the `DenseReLUMeta` structure also contains the following elements:
- `ξ` - Array with a length equal to the output dimension, specifying the expansion point of the Bernoulli distribution in the node. Default values are set to `ones(dim_out)`.

## Examples
The `DenseReLUMeta` structure is passed when defining the corresponding `DenseReLU` factor node:

```julia
y ~ DenseReLU(x, w, z, f) where { meta = DenseReLUMeta(3) }
```

In this example we specify that the random variable at the output has dimensionality 3. All other hyperparameters are set to their default values as specified above. Alternatively, these hyperparameters can be set manually when desired, for example:

```julia
y ~ DenseReLU(x, w, z, f) where { meta = DenseReLUMeta(3; β=100.0) }
```

"""
function DenseReLUMeta(dim_out::Int64; C::T1=1.0, β::T2=10.0, γ::T3=10.0) where { T1, T2, T3 }
    T = promote_type(T1, T2, T3, Float64)
    ξ = ones(T, dim_out)
    return DenseReLUMeta{T}(dim_out, C, β, γ, ξ)
end

getdimout(meta::DenseReLUMeta)          = meta.dim_out
getC(meta::DenseReLUMeta)               = meta.C
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

function ReactiveMP.interfaceindex(factornode::DenseReLUNode, iname::Symbol)
    iindex = findfirst(interface -> name(interface) === iname, interfaces(factornode))
    return iindex !== nothing ? iindex : error("Unknown interface ':$(iname)' for $(functionalform(factornode)) node")
end

# FreeEnergy related functions

@average_energy DenseReLU (q_output::NormalDistributionsFamily, q_input::NormalDistributionsFamily, q_w::NTuple{N, NormalDistributionsFamily}, q_z::NTuple{N, Bernoulli}, q_f::NTuple{N, UnivariateNormalDistributionsFamily}, meta::DenseReLUMeta) where N = begin

    # fetch statistics once
    my, vy      = mean_cov(q_output)
    mx, vx      = mean_cov(q_input)

    # fetch meta data once
    C           = getC(meta)
    β           = getβ(meta)
    γ           = getγ(meta)

    # average energy calculation
    U =  mapreduce(+, 1:N, init = 0.0) do k

        # fetch statistics per round
        mw, vw      = mean_cov(q_w[k])
        pz          = mean(q_z[k])
        mf, vf      = mean_cov(q_f[k])

        # fetch meta data per round
        ξk          = getξk(meta, k)

        # perform calculation
        return  0.5 * (log2π - log(γ) + γ * (vy[k] + pz*vf + abs2(my[k] - pz*mf))) +
                0.5 * (log2π - log(β) + β * (vf + mf^2 - 2*mf*dot(mw, mx) + tr( (mx*mx' + vx)*(mw*mw' + vw) ))) +
                0.5 * (C*mf + ξk) - pz*mf*C - log(sigmoid(ξk)) + (sigmoid(ξk) - 0.5)/2/ξk*(C^2*mf - ξk^2)

    end

end

function score(::Type{T}, objective::BetheFreeEnergy, ::FactorBoundFreeEnergy, ::Stochastic, node::DenseReLUNode{N, MeanField}, scheduler) where { T <: InfCountingReal, N }
    
    skip_strategy = marginal_skip_strategy(objective)

    stream = combineLatest((
        getmarginal(connectedvar(node.output), skip_strategy) |> schedule_on(scheduler),
        getmarginal(connectedvar(node.input), skip_strategy) |> schedule_on(scheduler),
        combineLatest(map((w) -> getmarginal(connectedvar(w), skip_strategy) |> schedule_on(scheduler), node.w), PushNew()),
        combineLatest(map((z) -> getmarginal(connectedvar(z), skip_strategy) |> schedule_on(scheduler), node.z), PushNew()),
        combineLatest(map((f) -> getmarginal(connectedvar(f), skip_strategy) |> schedule_on(scheduler), node.f), PushNew())
    ), PushNew())

    mapping = let fform = functionalform(node), meta = metadata(node)
        (marginals) -> begin 
            average_energy      = score(AverageEnergy(), fform, Val{ (:output, :input, :w, :z, :f) }, marginals, meta)

            output_entropy      = score(DifferentialEntropy(), marginals[1])
            input_entropy       = score(DifferentialEntropy(), marginals[2])
            w_entropies         = mapreduce((w) -> score(DifferentialEntropy(), w), +, marginals[3])
            z_entropies         = mapreduce((z) -> score(DifferentialEntropy(), z), +, marginals[4])
            f_entropies         = mapreduce((f) -> score(DifferentialEntropy(), f), +, marginals[5])

            return convert(T, average_energy - (output_entropy + input_entropy + w_entropies + z_entropies + f_entropies))
        end
    end

    return stream |> map(T, mapping)
end

as_node_functional_form(::Type{ <: DenseReLU }) = ValidNodeFunctionalForm()

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
