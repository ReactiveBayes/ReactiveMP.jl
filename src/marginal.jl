export Marginal, getdata, is_clamped, is_initial, as_marginal
export InitialMarginal, MarginalOrInitialMarginal
export SkipClamped, SkipInitial, SkipClampedAndInitial, IncludeAll

using Distributions
using Rocket

import Rocket: getrecent
import Base: ndims, precision, length, size

struct Marginal{D}
    data       :: D
    is_clamped :: Bool
    is_initial :: Bool
end

Base.show(io::IO, marginal::Marginal) = print(io, string("Marginal(", getdata(marginal), ")"))

getdata(marginal::Marginal)    = marginal.data
is_clamped(marginal::Marginal) = marginal.is_clamped
is_initial(marginal::Marginal) = marginal.is_initial

## Statistics 

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

## Utility functions

getdata(marginals::NTuple{ N, <: Marginal }) where N = map(getdata, marginals)

# as_marginal(distribution::Distribution) = Marginal(distribution)
# as_marginal(ntuple::NamedTuple)         = Marginal(ntuple)
as_marginal(marginal::Marginal)         = marginal

foldl_reduce_to_marginal(messages) = as_marginal(mapfoldl(as_message, *, messages))
foldr_reduce_to_marginal(messages) = as_marginal(mapfoldr(as_message, *, messages))
all_reduce_to_marginal(messages)   = as_marginal(prod_all(map(as_message, messages)))

prod_all(messages) = foldl(*, messages)

## Marginal observable

abstract type MarginalSkipStrategy end

struct SkipClamped           <: MarginalSkipStrategy end
struct SkipInitial           <: MarginalSkipStrategy end
struct SkipClampedAndInitial <: MarginalSkipStrategy end
struct IncludeAll            <: MarginalSkipStrategy end

struct MarginalObservable <: Subscribable{ Marginal }
    subject :: Rocket.RecentSubjectInstance{ Marginal, Subject{ Marginal, AsapScheduler, AsapScheduler } }
    stream  :: LazyObservable{ Marginal }
end

MarginalObservable() = MarginalObservable(RecentSubject(Marginal), lazy(Marginal))   

as_marginal_observable(observable::MarginalObservable, ::SkipClamped)           = observable |> filter(v -> !is_clamped(v))
as_marginal_observable(observable::MarginalObservable, ::SkipInitial)           = observable |> filter(v -> !is_initial(v))
as_marginal_observable(observable::MarginalObservable, ::SkipClampedAndInitial) = observable |> filter(v -> !is_initial(v))
as_marginal_observable(observable::MarginalObservable, ::IncludeAll)            = observable

as_marginal_observable(observable) = as_marginal_observable(observable, IncludeAll())

function as_marginal_observable(observable, skip_strategy::MarginalSkipStrategy)
    output = MarginalObservable()
    connect!(output, observable)
    return as_marginal_observable(output, skip_strategy)
end

Rocket.getrecent(observable::MarginalObservable) = Rocket.getrecent(observable.subject)
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
    next!(marginal.subject, Marginal(value, false, true))
    return nothing
end
