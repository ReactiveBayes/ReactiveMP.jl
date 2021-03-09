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

## Marginal

Distributions.mean(marginal::Marginal)      = Distributions.mean(getdata(marginal))
Distributions.median(marginal::Marginal)    = Distributions.median(getdata(marginal))
Distributions.mode(marginal::Marginal)      = Distributions.mode(getdata(marginal))
Distributions.shape(marginal::Marginal)     = Distributions.shape(getdata(marginal))
Distributions.scale(marginal::Marginal)     = Distributions.scale(getdata(marginal))
Distributions.rate(marginal::Marginal)      = Distributions.rate(getdata(marginal))
Distributions.var(marginal::Marginal)       = Distributions.var(getdata(marginal))
Distributions.std(marginal::Marginal)       = Distributions.std(getdata(marginal))
Distributions.cov(marginal::Marginal)       = Distributions.cov(getdata(marginal))
Distributions.invcov(marginal::Marginal)    = Distributions.invcov(getdata(marginal))
Distributions.logdetcov(marginal::Marginal) = Distributions.logdetcov(getdata(marginal))
Distributions.entropy(marginal::Marginal)   = Distributions.entropy(getdata(marginal))
Distributions.params(marginal::Marginal)    = Distributions.params(getdata(marginal))

Distributions.pdf(marginal::Marginal, x)    = Distributions.pdf(getdata(marginal), x)
Distributions.logpdf(marginal::Marginal, x) = Distributions.logpdf(getdata(marginal), x)

Base.precision(marginal::Marginal) = precision(getdata(marginal))
Base.ndims(marginal::Marginal)     = ndims(getdata(marginal))
Base.length(marginal::Marginal)    = length(getdata(marginal))
Base.size(marginal::Marginal)      = size(getdata(marginal))

probvec(marginal::Marginal)         = probvec(getdata(marginal))
weightedmean(marginal::Marginal)    = weightedmean(getdata(marginal))
inversemean(marginal::Marginal)     = inversemean(getdata(marginal))
logmean(marginal::Marginal)         = logmean(getdata(marginal))
meanlogmean(marginal::Marginal)     = meanlogmean(getdata(marginal))
mirroredlogmean(marginal::Marginal) = mirroredlogmean(getdata(marginal))
loggammamean(marginal::Marginal)    = loggammamean(getdata(marginal))

## Initial Marginal

struct InitialMarginal{D}
    data :: D
end

Base.show(io::IO, marginal::InitialMarginal) = print(io, string("InitialMarginal(", getdata(marginal), ")"))

getdata(marginal::InitialMarginal) = marginal.data

as_marginal(imarginal::InitialMarginal) = Marginal(getdata(imarginal))

## Utility functions

const MarginalOrInitialMarginal = Union{ Marginal, InitialMarginal }

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

as_marginal_observable(::SkipInitial, observable::MarginalObservable)    = observable |> filter(r -> typeof(r) <: Marginal) |> map(Marginal, v -> v)
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
