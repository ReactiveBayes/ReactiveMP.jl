export Marginal, getdata, as_marginal

using Distributions
using Rocket

import Base: ndims, precision, length, size

struct Marginal{D}
    data :: D
end

getdata(marginal::Marginal)                          = marginal.data
getdata(marginals::NTuple{ N, <: Marginal }) where N = map(getdata, marginals)

## Marginal

Distributions.mean(marginal::Marginal)      = Distributions.mean(getdata(marginal))
Distributions.median(marginal::Marginal)    = Distributions.median(getdata(marginal))
Distributions.mode(marginal::Marginal)      = Distributions.mode(getdata(marginal))
Distributions.var(marginal::Marginal)       = Distributions.var(getdata(marginal))
Distributions.std(marginal::Marginal)       = Distributions.std(getdata(marginal))
Distributions.cov(marginal::Marginal)       = Distributions.cov(getdata(marginal))
Distributions.invcov(marginal::Marginal)    = Distributions.invcov(getdata(marginal))
Distributions.logdetcov(marginal::Marginal) = Distributions.logdetcov(getdata(marginal))
Distributions.entropy(marginal::Marginal)   = Distributions.entropy(getdata(marginal))

Distributions.pdf(marginal::Marginal, x)    = Distributions.pdf(getdata(marginal), x)
Distributions.logpdf(marginal::Marginal, x) = Distributions.logpdf(getdata(marginal), x)

Base.precision(marginal::Marginal) = precision(getdata(marginal))
Base.ndims(marginal::Marginal)     = ndims(getdata(marginal))
Base.length(marginal::Marginal)    = length(getdata(marginal))
Base.size(marginal::Marginal)      = size(getdata(marginal))

probvec(marginal::Marginal)         = probvec(getdata(marginal))
logmean(marginal::Marginal)         = logmean(getdata(marginal))
inversemean(marginal::Marginal)     = inversemean(getdata(marginal))
mirroredlogmean(marginal::Marginal) = mirroredlogmean(getdata(marginal))

## Utility functions

as_marginal(data)               = Marginal(data)
as_marginal(marginal::Marginal) = marginal

const __as_marginal_operator = Rocket.map(Marginal, as_marginal)

as_marginal() = __as_marginal_operator

function __reduce_to_marginal(messages)
    return as_marginal(reduce_messages(messages))
end

const reduce_to_marginal = Rocket.map(Marginal, __reduce_to_marginal)

## Marginal observable

struct MarginalObservable <: Subscribable{Marginal}
    subject :: Rocket.RecentSubjectInstance{ Marginal, Subject{ Marginal,AsapScheduler,AsapScheduler } }
    stream  :: LazyObservable{Marginal}
end

as_marginal_observable(observable::MarginalObservable) = observable

function as_marginal_observable(observable)
    output = MarginalObservable()
    connect!(output, observable)
    return output
end

MarginalObservable() = MarginalObservable(RecentSubject(Marginal), lazy(Marginal))

getrecent(observable::MarginalObservable) = Rocket.getrecent(observable.subject)
getrecent(observables::Tuple)             = getrecent.(observables)                       
getrecent(::Nothing)                      = nothing

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