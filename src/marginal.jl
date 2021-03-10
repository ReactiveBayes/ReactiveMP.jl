export Marginal, getdata, as_marginal
export InitialMarginal, MarginalOrInitialMarginal
export SkipInitial, IncludeInitial

using Distributions
using Rocket

import Rocket: getrecent
import Base: ndims, precision, length, size

struct Marginal{D}
    data :: D
end

Base.show(io::IO, marginal::Marginal) = print(io, string("Marginal(", getdata(marginal), ")"))

getdata(marginal::Marginal) = marginal.data

## Initial Marginal

struct InitialMarginal{D}
    data :: D
end

Base.show(io::IO, marginal::InitialMarginal) = print(io, string("InitialMarginal(", getdata(marginal), ")"))

getdata(marginal::InitialMarginal) = marginal.data

as_marginal(imarginal::InitialMarginal) = Marginal(getdata(imarginal))

## Aliases

const MarginalOrInitialMarginal = Union{ Marginal, InitialMarginal }

## Statistics 

Distributions.mean(marginal::MarginalOrInitialMarginal)      = Distributions.mean(getdata(marginal))
Distributions.median(marginal::MarginalOrInitialMarginal)    = Distributions.median(getdata(marginal))
Distributions.mode(marginal::MarginalOrInitialMarginal)      = Distributions.mode(getdata(marginal))
Distributions.shape(marginal::MarginalOrInitialMarginal)     = Distributions.shape(getdata(marginal))
Distributions.scale(marginal::MarginalOrInitialMarginal)     = Distributions.scale(getdata(marginal))
Distributions.rate(marginal::MarginalOrInitialMarginal)      = Distributions.rate(getdata(marginal))
Distributions.var(marginal::MarginalOrInitialMarginal)       = Distributions.var(getdata(marginal))
Distributions.std(marginal::MarginalOrInitialMarginal)       = Distributions.std(getdata(marginal))
Distributions.cov(marginal::MarginalOrInitialMarginal)       = Distributions.cov(getdata(marginal))
Distributions.invcov(marginal::MarginalOrInitialMarginal)    = Distributions.invcov(getdata(marginal))
Distributions.logdetcov(marginal::MarginalOrInitialMarginal) = Distributions.logdetcov(getdata(marginal))
Distributions.entropy(marginal::MarginalOrInitialMarginal)   = Distributions.entropy(getdata(marginal))
Distributions.params(marginal::MarginalOrInitialMarginal)    = Distributions.params(getdata(marginal))

Distributions.pdf(marginal::MarginalOrInitialMarginal, x)    = Distributions.pdf(getdata(marginal), x)
Distributions.logpdf(marginal::MarginalOrInitialMarginal, x) = Distributions.logpdf(getdata(marginal), x)

Base.precision(marginal::MarginalOrInitialMarginal) = precision(getdata(marginal))
Base.ndims(marginal::MarginalOrInitialMarginal)     = ndims(getdata(marginal))
Base.length(marginal::MarginalOrInitialMarginal)    = length(getdata(marginal))
Base.size(marginal::MarginalOrInitialMarginal)      = size(getdata(marginal))

probvec(marginal::MarginalOrInitialMarginal)         = probvec(getdata(marginal))
weightedmean(marginal::MarginalOrInitialMarginal)    = weightedmean(getdata(marginal))
inversemean(marginal::MarginalOrInitialMarginal)     = inversemean(getdata(marginal))
logmean(marginal::MarginalOrInitialMarginal)         = logmean(getdata(marginal))
meanlogmean(marginal::MarginalOrInitialMarginal)     = meanlogmean(getdata(marginal))
mirroredlogmean(marginal::MarginalOrInitialMarginal) = mirroredlogmean(getdata(marginal))
loggammamean(marginal::MarginalOrInitialMarginal)    = loggammamean(getdata(marginal))

## Utility functions

getdata(marginals::NTuple{ N, <: Marginal }) where N = map(getdata, marginals)

as_marginal(distribution::Distribution) = Marginal(distribution)
as_marginal(ntuple::NamedTuple)         = Marginal(ntuple)
as_marginal(marginal::Marginal)         = marginal

const __as_marginal_operator = Rocket.map(Marginal, as_marginal)

as_marginal() = __as_marginal_operator

reduce_to_marginal(messages) = foldl_reduce_to_marginal(messages)

foldl_reduce_to_marginal(messages) = as_marginal(mapfoldl(as_message, *, messages; init = Message(missing)))
foldr_reduce_to_marginal(messages) = as_marginal(mapfoldr(as_message, *, messages; init = Message(missing)))
all_reduce_to_marginal(messages)   = as_marginal(prod_all(map(as_message, messages)))

prod_all(messages) = foldl(*, messages; init = Message(missing))

## Marginal observable

struct SkipInitial end
struct IncludeInitial end

struct MarginalObservable <: Subscribable{ MarginalOrInitialMarginal }
    subject :: Rocket.RecentSubjectInstance{ MarginalOrInitialMarginal, Subject{ MarginalOrInitialMarginal, AsapScheduler, AsapScheduler } }
    stream  :: LazyObservable{ MarginalOrInitialMarginal }
end

MarginalObservable() = MarginalObservable(RecentSubject(MarginalOrInitialMarginal), lazy(MarginalOrInitialMarginal))

as_marginal_observable(::SkipInitial, observable::MarginalObservable)    = observable |> filter_type(Marginal)
as_marginal_observable(::IncludeInitial, observable::MarginalObservable) = observable

function as_marginal_observable(skip_strategy::Union{ SkipInitial, IncludeInitial }, observable)
    output = MarginalObservable()
    connect!(output, observable)
    return as_marginal_observable(skip_strategy, output)
end

Rocket.getrecent(observable::MarginalObservable) = as_marginal(Rocket.getrecent(observable.subject))
Rocket.getrecent(observables::Tuple)             = Rocket.getrecent.(observables)
Rocket.getrecent(::Nothing)                      = nothing

function Rocket.on_subscribe!(observable::MarginalObservable, actor)
    return subscribe!(observable.stream, actor)
end

function connect!(marginal::MarginalObservable, source)
    set!(marginal.stream, source |> multicast(marginal.subject) |> ref_count())
    return nothing
end

function setmarginal!(marginal::MarginalObservable, value)
    next!(marginal.subject, InitialMarginal(value))
    return nothing
end
