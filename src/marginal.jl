export Marginal, getdata, as_marginal

using Distributions
using Rocket

import Base: ndims, precision

struct Marginal{D}
    data :: D
end

getdata(marginal::Marginal) = marginal.data

## Marginal

Distributions.mean(marginal::Marginal) = Distributions.mean(getdata(marginal))
Distributions.var(marginal::Marginal)  = Distributions.var(getdata(marginal))
Distributions.std(marginal::Marginal)  = Distributions.std(getdata(marginal))
Distributions.cov(marginal::Marginal)  = Distributions.cov(getdata(marginal))

Base.precision(marginal::Marginal) = precision(getdata(marginal))
Base.ndims(marginal::Marginal)     = ndims(getdata(marginal))

## Delta-function marginal
 
Distributions.mean(marginal::Marginal{T}) where { T <: Real } = getdata(marginal)
Distributions.var(marginal::Marginal{T}) where { T <: Real }  = zero(T)
Distributions.std(marginal::Marginal{T}) where { T <: Real }  = zero(T)
Distributions.cov(marginal::Marginal{T}) where { T <: Real }  = zero(T)

Base.precision(marginal::Marginal{T}) where { T <: Real } = Inf
Base.ndims(marginal::Marginal{T})     where { T <: Real } = 1

logmean(marginal::Marginal{T}) where { T <: Real }     = log(getdata(marginal))
inversemean(marginal::Marginal{T}) where { T <: Real } = 1.0 / getdata(marginal)

## Vector-based delta function marginal

Distributions.mean(marginal::Marginal{V}) where { T, V <: Vector{T} } = getdata(marginal)
Distributions.var(marginal::Marginal{V})  where { T, V <: Vector{T} } = zeros(T, (ndims(marginal), ))
Distributions.std(marginal::Marginal{V})  where { T, V <: Vector{T} } = zeros(T, (ndims(marginal), ))
Distributions.cov(marginal::Marginal{V})  where { T, V <: Vector{T} } = zeros(T, (ndims(marginal), ndims(marginal)))

Base.precision(marginal::Marginal{T}) where { T <: Vector } = Inf
Base.ndims(marginal::Marginal{T})     where { T <: Vector } = length(getdata(marginal))

## Utility functions

as_marginal(data)               = Marginal(data)
as_marginal(marginal::Marginal) = marginal

const __as_marginal_operator = Rocket.map(Marginal, as_marginal)

as_marginal() = __as_marginal_operator

const reduce_to_marginal = Rocket.map(Marginal, (messages) -> as_marginal(reduce_messages(messages)))

## Marginal observable

struct MarginalObservable <: Subscribable{Marginal}
    subject :: Rocket.ReplaySubjectInstance{ Marginal, Subject{ Marginal,AsapScheduler,AsapScheduler } }
    stream  :: LazyObservable{Marginal}
end

MarginalObservable() = MarginalObservable(ReplaySubject(Marginal, 1), lazy(Marginal))

function Rocket.on_subscribe!(observable::MarginalObservable, actor)
    return subscribe!(observable.stream, actor)
end

function connect!(marginal::MarginalObservable, source)
    set!(marginal.stream, source |> multicast(marginal.subject) |> ref_count())
    return nothing
end

function setmarginal!(marginal::MarginalObservable, value)
    next!(marginal.subject, as_marginal(value))
    return nothing
end