export Marginal, getdata, is_clamped, is_initial, as_marginal
export SkipClamped, SkipInitial, SkipClampedAndInitial, IncludeAll

using Distributions
using Rocket

import Rocket: getrecent
import Base: ==, ndims, precision, length, size, iterate

"""
    Marginal(data, is_clamped, is_initial, addons)

An implementation of a marginal in variational message passing framework.

# Arguments
- `data::D`: marginal always holds some data object associated with it, which is usually a probability distribution
- `is_clamped::Bool`, specifies if this marginal was the result of constant computations (e.g. clamped constants)
- `is_initial::Bool`, specifies if this marginal was used for initialization
- `addons::A`, specifies the addons of the marginal, which may carry extra bits of information, e.g. debug information, memory, etc.

# Example 

```jldoctest
julia> distribution = Gamma(10.0, 2.0)
Distributions.Gamma{Float64}(α=10.0, θ=2.0)

julia> message = Marginal(distribution, false, true, nothing)
Marginal(Distributions.Gamma{Float64}(α=10.0, θ=2.0))

julia> mean(message) 
20.0

julia> getdata(message)
Distributions.Gamma{Float64}(α=10.0, θ=2.0)

julia> is_clamped(message)
false

julia> is_initial(message)
true
```
"""
mutable struct Marginal{D, A}  # `mutable` structure here appears to be more performance 
    const data       :: D      # in `RxInfer` benchmarks
    const is_clamped :: Bool   # could be revised at some point though
    const is_initial :: Bool
    const addons     :: A
end

function Base.show(io::IO, marginal::Marginal)
    print(io, string("Marginal(", getdata(marginal), ")"))
    if !isnothing(getaddons(marginal))
        print(io, ") with ", string(getaddons(marginal)))
    end
end

function Base.:(==)(left::Marginal, right::Marginal)
    # We need this dummy method as Julia is not smart enough to 
    # do that automatically if `data` is mutable
    return left.is_clamped == right.is_clamped && left.is_initial == right.is_initial && left.data == right.data && left.addons == right.addons
end

"""
    getdata(marginal::Marginal)    

Returns `data` associated with the `marginal`.
"""
getdata(marginal::Marginal) = marginal.data

"""
    is_clamped(marginal::Marginal)

Checks if `marginal` is clamped or not.

See also: [`is_initial`](@ref)
"""
is_clamped(marginal::Marginal) = marginal.is_clamped

"""
    is_initial(marginal::Marginal)

Checks if `marginal` is initial or not.

See also: [`is_clamped`](@ref)
"""
is_initial(marginal::Marginal) = marginal.is_initial

"""
    getaddons(marginal::Marginal)

Returns `addons` associated with the `marginal`.
"""
getaddons(marginal::Marginal) = marginal.addons

typeofdata(marginal::Marginal) = typeof(getdata(marginal))

getdata(marginals::NTuple{N, <:Marginal}) where {N} = map(getdata, marginals)
getdata(marginals::AbstractArray{<:Marginal})       = map(getdata, marginals)

ispointmass(marginal::Marginal) = ispointmass(marginal, getdata(marginal))
ispointmass(marginal::Marginal, data::PointMass) = true
ispointmass(marginal::Marginal, data) = false

## Statistics 

Distributions.pdf(marginal::Marginal, x)    = Distributions.pdf(getdata(marginal), x)
Distributions.logpdf(marginal::Marginal, x) = Distributions.logpdf(getdata(marginal), x)

MacroHelpers.@proxy_methods Marginal getdata [
    BayesBase.mean,
    BayesBase.median,
    BayesBase.mode,
    BayesBase.shape,
    BayesBase.scale,
    BayesBase.rate,
    BayesBase.var,
    BayesBase.std,
    BayesBase.cov,
    BayesBase.invcov,
    BayesBase.logdetcov,
    BayesBase.entropy,
    BayesBase.params,
    BayesBase.mean_cov,
    BayesBase.mean_var,
    BayesBase.mean_invcov,
    BayesBase.mean_precision,
    BayesBase.weightedmean_cov,
    BayesBase.weightedmean_var,
    BayesBase.weightedmean_invcov,
    BayesBase.weightedmean_precision,
    BayesBase.probvec,
    BayesBase.weightedmean,
    Base.precision,
    Base.length,
    Base.ndims,
    Base.size
]

# Eltype is special here, because it should be only defined on types
# Otherwise it causes invalidations and slower compile times
Base.eltype(::Type{<:Marginal{D}}) where {D} = Base.eltype(D)

Distributions.mean(fn::Function, marginal::Marginal) = mean(fn, getdata(marginal))

"""
    as_marginal(any)

A function that converts `any` to an instance of `Marginal`.
"""
function as_marginal end

as_marginal(marginal::Marginal) = marginal

dropproxytype(::Type{<:Marginal{T}}) where {T} = T

## Marginal observable

abstract type MarginalSkipStrategy end

struct SkipClamped <: MarginalSkipStrategy end
struct SkipInitial <: MarginalSkipStrategy end
struct SkipClampedAndInitial <: MarginalSkipStrategy end
struct IncludeAll <: MarginalSkipStrategy end

Base.broadcastable(::SkipClamped) = Ref(SkipClamped())
Base.broadcastable(::SkipInitial) = Ref(SkipInitial())
Base.broadcastable(::SkipClampedAndInitial) = Ref(SkipClampedAndInitial())
Base.broadcastable(::IncludeAll) = Ref(IncludeAll())

apply_skip_filter(observable, ::SkipClamped)           = observable |> filter(v -> !is_clamped(v))
apply_skip_filter(observable, ::SkipInitial)           = observable |> filter(v -> !is_initial(v))
apply_skip_filter(observable, ::SkipClampedAndInitial) = observable |> filter(v -> !is_initial(v) && !is_clamped(v))
apply_skip_filter(observable, ::IncludeAll)            = observable

struct MarginalObservable <: Subscribable{Marginal}
    subject :: Rocket.RecentSubjectInstance{Marginal, Subject{Marginal, AsapScheduler, AsapScheduler}}
    stream  :: LazyObservable{Marginal}
end

MarginalObservable() = MarginalObservable(RecentSubject(Marginal), lazy(Marginal))

as_marginal_observable(observable::MarginalObservable, skip_strategy::MarginalSkipStrategy) = apply_skip_filter(observable, skip_strategy)
as_marginal_observable(observable)                                                          = as_marginal_observable(observable, IncludeAll())

function as_marginal_observable(observable, skip_strategy::MarginalSkipStrategy)
    output = MarginalObservable()
    connect!(output, observable)
    return as_marginal_observable(output, skip_strategy)
end

Rocket.getrecent(observable::MarginalObservable) = Rocket.getrecent(observable.subject)

@inline Rocket.on_subscribe!(observable::MarginalObservable, actor) = subscribe!(observable.stream, actor)

@inline Rocket.subscribe!(observable::MarginalObservable, actor::Rocket.Actor{<:Marginal})           = Rocket.on_subscribe!(observable.stream, actor)
@inline Rocket.subscribe!(observable::MarginalObservable, actor::Rocket.NextActor{<:Marginal})       = Rocket.on_subscribe!(observable.stream, actor)
@inline Rocket.subscribe!(observable::MarginalObservable, actor::Rocket.ErrorActor{<:Marginal})      = Rocket.on_subscribe!(observable.stream, actor)
@inline Rocket.subscribe!(observable::MarginalObservable, actor::Rocket.CompletionActor{<:Marginal}) = Rocket.on_subscribe!(observable.stream, actor)

@inline Rocket.subscribe!(observable::MarginalObservable, actor::Rocket.Subject{<:Marginal})                 = Rocket.on_subscribe!(observable.stream, actor)
@inline Rocket.subscribe!(observable::MarginalObservable, actor::Rocket.BehaviorSubjectInstance{<:Marginal}) = Rocket.on_subscribe!(observable.stream, actor)
@inline Rocket.subscribe!(observable::MarginalObservable, actor::Rocket.PendingSubjectInstance{<:Marginal})  = Rocket.on_subscribe!(observable.stream, actor)
@inline Rocket.subscribe!(observable::MarginalObservable, actor::Rocket.RecentSubjectInstance{<:Marginal})   = Rocket.on_subscribe!(observable.stream, actor)
@inline Rocket.subscribe!(observable::MarginalObservable, actor::Rocket.ReplaySubjectInstance{<:Marginal})   = Rocket.on_subscribe!(observable.stream, actor)

function connect!(marginal::MarginalObservable, source)
    set!(marginal.stream, source |> multicast(marginal.subject) |> ref_count())
    return nothing
end

function setmarginal!(marginal::MarginalObservable, value)
    next!(marginal.subject, Marginal(value, false, true, nothing))
    return nothing
end

## Marginal Mapping structure
## https://github.com/JuliaLang/julia/issues/42559
## Explanation: Julia cannot fully infer type of the lambda callback function in activate! method in node.jl file
## We create a lambda-like callable structure to improve type inference and make it more stable
## However it is not fully inferrable due to dynamic tags and variable constraints, but still better than just a raw lambda callback

struct MarginalMapping{F, T, N, M, A, R}
    vtag            :: T
    msgs_names      :: N
    marginals_names :: M
    meta            :: A
    factornode      :: R
end

marginal_mapping_fform(::MarginalMapping{F}) where {F} = F
marginal_mapping_fform(::MarginalMapping{F}) where {F <: Function} = F.instance

function MarginalMapping(::Type{F}, vtag::T, msgs_names::N, marginals_names::M, meta::A, factornode::R) where {F, T, N, M, A, R}
    return MarginalMapping{F, T, N, M, A, R}(vtag, msgs_names, marginals_names, meta, factornode)
end

function MarginalMapping(::F, vtag::T, msgs_names::N, marginals_names::M, meta::A, factornode::R) where {F <: Function, T, N, M, A, R}
    return MarginalMapping{F, T, N, M, A, R}(vtag, msgs_names, marginals_names, meta, factornode)
end

function (mapping::MarginalMapping)(dependencies)
    messages  = getrecent(dependencies[1])
    marginals = getrecent(dependencies[2])

    # Marginal is clamped if all of the inputs are clamped
    is_marginal_clamped = __check_all(is_clamped, messages) && __check_all(is_clamped, marginals)

    # Marginal is initial if it is not clamped and all of the inputs are either clamped or initial
    is_marginal_initial = !is_marginal_clamped && (__check_all(is_clamped_or_initial, messages) && __check_all(is_clamped_or_initial, marginals))

    marginal = if !isnothing(messages) && any(ismissing, TupleTools.flatten(getdata.(messages)))
        missing
    elseif !isnothing(marginals) && any(ismissing, TupleTools.flatten(getdata.(marginals)))
        missing
    else
        marginalrule(marginal_mapping_fform(mapping), mapping.vtag, mapping.msgs_names, messages, mapping.marginals_names, marginals, mapping.meta, mapping.factornode)
    end

    return Marginal(marginal, is_marginal_clamped, is_marginal_initial, nothing)
end

Base.map(::Type{T}, mapping::M) where {T, M <: MarginalMapping} = Rocket.MapOperator{T, M}(mapping)
