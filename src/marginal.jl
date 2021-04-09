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

# TupleTools.prod is a more efficient version of Base.all for NTuple here
is_clamped(marginals::NTuple{ N, <: Marginal }) where N = TupleTools.prod(map(is_clamped, marginals))
is_initial(marginals::NTuple{ N, <: Marginal }) where N = TupleTools.prod(map(is_initial, marginals))

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

getdata(marginals::NTuple{ N, <: Marginal })      where N = map(getdata, marginals)
getdata(marginals::AbstractVector{ <: Marginal })         = map(getdata, marginals) 

as_marginal(marginal::Marginal) = marginal

# Note: we need extra Base.Generator(as_message, messages) step here, because some of the messages might be VMP messages
# We want to cast it explicitly to a Message structure (which as_message does in case of VariationalMessage)
# We use with Base.Generator to reduce an amount of memory used by this procedure since Generator generates items lazily
foldl_reduce_to_marginal(prod_parametrisation) = (messages) -> as_marginal(foldl((left, right) -> multiply_messages(prod_parametrisation, left, right), Base.Generator(as_message, messages)))
foldr_reduce_to_marginal(prod_parametrisation) = (messages) -> as_marginal(foldr((left, right) -> multiply_messages(prod_parametrisation, left, right), Base.Generator(as_message, messages)))

function all_reduce_to_marginal(prod_parametrisation) 
    return let prod_parametrisation = prod_parametrisation
        (messages) -> begin
            # We propagate clamped message, in case if both are clamped
            is_prod_clamped = __check_all(is_clamped, messages)
            # We propagate initial message, in case if both are initial or left is initial and right is clameped or vice-versa
            is_prod_initial = !is_prod_clamped && __check_all(v -> is_clamped(v) || is_initial(v), messages)
            return Marginal(prod_all(prod_parametrisation, Base.Generator(getdata, messages)), is_prod_clamped, is_prod_initial)
        end
    end
end

# Fallback option
prod_all(prod_parametrisation, inputs) = foldl((left, right) -> prod(prod_parametrisation, left, right), inputs)

## Marginal observable

abstract type MarginalSkipStrategy end

struct SkipClamped           <: MarginalSkipStrategy end
struct SkipInitial           <: MarginalSkipStrategy end
struct SkipClampedAndInitial <: MarginalSkipStrategy end
struct IncludeAll            <: MarginalSkipStrategy end

apply_skip_filter(observable, ::SkipClamped)           = observable |> filter(v -> !is_clamped(v))
apply_skip_filter(observable, ::SkipInitial)           = observable |> filter(v -> !is_initial(v))
apply_skip_filter(observable, ::SkipClampedAndInitial) = observable |> filter(v -> !is_initial(v) && !is_clamped(v))
apply_skip_filter(observable, ::IncludeAll)            = observable

struct MarginalObservable <: Subscribable{ Marginal }
    subject :: Rocket.RecentSubjectInstance{ Marginal, Subject{ Marginal, AsapScheduler, AsapScheduler } }
    stream  :: LazyObservable{ Marginal }
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
Rocket.getrecent(observables::Tuple)             = Rocket.getrecent.(observables)
Rocket.getrecent(observables::AbstractVector)    = Rocket.getrecent.(observables)
Rocket.getrecent(::Nothing)                      = nothing

@inline Rocket.on_subscribe!(observable::MarginalObservable, actor) = subscribe!(observable.stream, actor)

@inline Rocket.subscribe!(observable::MarginalObservable, actor::Rocket.Actor{ <: Marginal })           = Rocket.on_subscribe!(observable.stream, actor)
@inline Rocket.subscribe!(observable::MarginalObservable, actor::Rocket.NextActor{ <: Marginal })       = Rocket.on_subscribe!(observable.stream, actor)
@inline Rocket.subscribe!(observable::MarginalObservable, actor::Rocket.ErrorActor{ <: Marginal })      = Rocket.on_subscribe!(observable.stream, actor)
@inline Rocket.subscribe!(observable::MarginalObservable, actor::Rocket.CompletionActor{ <: Marginal }) = Rocket.on_subscribe!(observable.stream, actor)

@inline Rocket.subscribe!(observable::MarginalObservable, actor::Rocket.Subject{ <: Marginal })                 = Rocket.on_subscribe!(observable.stream, actor)
@inline Rocket.subscribe!(observable::MarginalObservable, actor::Rocket.BehaviorSubjectInstance{ <: Marginal }) = Rocket.on_subscribe!(observable.stream, actor)
@inline Rocket.subscribe!(observable::MarginalObservable, actor::Rocket.PendingSubjectInstance{ <: Marginal })  = Rocket.on_subscribe!(observable.stream, actor)
@inline Rocket.subscribe!(observable::MarginalObservable, actor::Rocket.RecentSubjectInstance{ <: Marginal })   = Rocket.on_subscribe!(observable.stream, actor)
@inline Rocket.subscribe!(observable::MarginalObservable, actor::Rocket.ReplaySubjectInstance{ <: Marginal })   = Rocket.on_subscribe!(observable.stream, actor)

function connect!(marginal::MarginalObservable, source)
    set!(marginal.stream, source |> multicast(marginal.subject) |> ref_count())
    return nothing
end

function setmarginal!(marginal::MarginalObservable, value)
    next!(marginal.subject, Marginal(value, false, true))
    return nothing
end
