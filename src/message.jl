export Message, getdata, is_clamped, is_initial, as_message
export multiply_messages

using Distributions
using Rocket

import Rocket: getrecent
import Base: *, +, ndims, precision, length, size, show

abstract type AbstractMessage end

struct Message{D} <: AbstractMessage
    data       :: D
    is_clamped :: Bool
    is_initial :: Bool
end

getdata(message::Message)    = message.data
is_clamped(message::Message) = message.is_clamped
is_initial(message::Message) = message.is_initial

getdata(messages::NTuple{ N, <: Message })    where N = map(getdata, messages)

materialize!(message::Message) = message

Base.show(io::IO, message::Message) = print(io, string("Message(", getdata(message), ")"))

## Message

Base.:*(left::Message, right::Message) = multiply_messages(ProdPreserveParametrisation(), left, right)

function multiply_messages(prod_parametrisation, left::Message, right::Message) 
    # We propagate clamped message, in case if both are clamped
    is_prod_clamped = is_clamped(left) && is_clamped(right)
    # We propagate initial message, in case if both are initial or left is initial and right is clameped or vice-versa
    is_prod_initial = !is_prod_clamped && (is_initial(left) || is_clamped(left)) && (is_initial(right) || is_clamped(right))

    return Message(prod(prod_parametrisation, getdata(left), getdata(right)), is_prod_clamped, is_prod_initial)
end

# Base.:*(m1::Message, m2::Message) = multiply_messages(m1, m2)

Distributions.pdf(message::Message, x)    = Distributions.pdf(getdata(message), x)
Distributions.logpdf(message::Message, x) = Distributions.logpdf(getdata(message), x)

MacroHelpers.@proxy_methods Message getdata [
    Distributions.mean,
    Distributions.median,
    Distributions.mode,
    Distributions.shape,
    Distributions.scale,
    Distributions.rate,
    Distributions.var,
    Distributions.std,
    Distributions.cov,
    Distributions.invcov,
    Distributions.logdetcov,
    Distributions.entropy,
    Distributions.params,
    Base.precision,
    Base.length,
    Base.ndims,
    Base.size,
    probvec,
    weightedmean,
    inversemean,
    logmean,
    meanlogmean,
    mirroredlogmean,
    loggammamean
]

## Variational Message

mutable struct VariationalMessage{R, S, F} <: AbstractMessage
    messages   :: R
    marginals  :: S
    mappingFn  :: F
    cache      :: Union{Nothing, Message}
end

VariationalMessage(messages::R, marginals::S, mappingFn::F) where { R, S, F } = VariationalMessage(messages, marginals, mappingFn, nothing)

Base.show(io::IO, ::VariationalMessage) = print(io, string("VariationalMessage(:postponed)"))

getcache(vmessage::VariationalMessage)                    = vmessage.cache
setcache!(vmessage::VariationalMessage, message::Message) = vmessage.cache = message

compute_message(vmessage::VariationalMessage) = vmessage.mappingFn((vmessage.messages, getrecent(vmessage.marginals)))

function materialize!(vmessage::VariationalMessage)
    cache = getcache(vmessage)
    if cache !== nothing
        return cache
    end
    message = compute_message(vmessage)
    setcache!(vmessage, message)
    return message
end

## Messages Product

"""
    MessagesProduct

If inference backend cannot return an analytical solution for a product of two messages it may fallback to MessagesProduct structure
`MessagesProduct` is useful to propagate the exact forms of two messages until it hits some approximation method for form-constraint.
However `MessagesProduct` cannot be used to compute statistics such as mean or variance. 
It has to be approximated before using in actual inference procedure.

Backend exploits form constraints specification which usually help to deal with intractable messages products. 
User may use EM form constraint with a specific optimisation algorithm or it may approximate intractable product with Gaussian Distribution
using for example Laplace approximation 

See also: [`prod`](@ref)
"""
struct MessagesProduct{ L, R }
    left  :: L
    right :: R
end

Distributions.mean(product::MessagesProduct)      = error("mean() is not defined for $(product). MessagesProduct structure has to be approximated and cannot be used in inference procedure.")
Distributions.median(product::MessagesProduct)    = error("median() is not defined for $(product). MessagesProduct structure has to be approximated and cannot be used in inference procedure.")
Distributions.mode(product::MessagesProduct)      = error("mode() is not defined for $(product). MessagesProduct structure has to be approximated and cannot be used in inference procedure.")
Distributions.shape(product::MessagesProduct)     = error("shape() is not defined for $(product). MessagesProduct structure has to be approximated and cannot be used in inference procedure.")
Distributions.scale(product::MessagesProduct)     = error("scale() is not defined for $(product). MessagesProduct structure has to be approximated and cannot be used in inference procedure.")
Distributions.rate(product::MessagesProduct)      = error("rate() is not defined for $(product). MessagesProduct structure has to be approximated and cannot be used in inference procedure.")
Distributions.var(product::MessagesProduct)       = error("var() is not defined for $(product). MessagesProduct structure has to be approximated and cannot be used in inference procedure.")
Distributions.std(product::MessagesProduct)       = error("std() is not defined for $(product). MessagesProduct structure has to be approximated and cannot be used in inference procedure.")
Distributions.cov(product::MessagesProduct)       = error("cov() is not defined for $(product). MessagesProduct structure has to be approximated and cannot be used in inference procedure.")
Distributions.invcov(product::MessagesProduct)    = error("invcov() is not defined for $(product). MessagesProduct structure has to be approximated and cannot be used in inference procedure.")
Distributions.logdetcov(product::MessagesProduct) = error("logdetcov() is not defined for $(product). MessagesProduct structure has to be approximated and cannot be used in inference procedure.")
Distributions.entropy(product::MessagesProduct)   = error("entropy() is not defined for $(product). MessagesProduct structure has to be approximated and cannot be used in inference procedure.")
Distributions.params(product::MessagesProduct)    = error("params() is not defined for $(product). MessagesProduct structure has to be approximated and cannot be used in inference procedure.")

Distributions.pdf(product::MessagesProduct, x)    = Distributions.pdf(product.left, x) * Distributions.pdf(product.right, x)
Distributions.logpdf(product::MessagesProduct, x) = Distributions.logpdf(product.left, x) + Distributions.logpdf(product.right, x)

Base.precision(product::MessagesProduct) = error("precision() is not defined for $(product). MessagesProduct structure has to be approximated and cannot be used in inference procedure.")
Base.length(product::MessagesProduct)    = error("length() is not defined for $(product). MessagesProduct structure has to be approximated and cannot be used in inference procedure.")
Base.ndims(product::MessagesProduct)     = error("ndims() is not defined for $(product). MessagesProduct structure has to be approximated and cannot be used in inference procedure.")
Base.size(product::MessagesProduct)      = error("size() is not defined for $(product). MessagesProduct structure has to be approximated and cannot be used in inference procedure.")

probvec(product::MessagesProduct)         = error("probvec() is not defined for $(product). MessagesProduct structure has to be approximated and cannot be used in inference procedure.")
weightedmean(product::MessagesProduct)    = error("weightedmean() is not defined for $(product). MessagesProduct structure has to be approximated and cannot be used in inference procedure.")
inversemean(product::MessagesProduct)     = error("inversemean() is not defined for $(product). MessagesProduct structure has to be approximated and cannot be used in inference procedure.")
logmean(product::MessagesProduct)         = error("logmean() is not defined for $(product). MessagesProduct structure has to be approximated and cannot be used in inference procedure.")
meanlogmean(product::MessagesProduct)     = error("meanlogmean() is not defined for $(product). MessagesProduct structure has to be approximated and cannot be used in inference procedure.")
mirroredlogmean(product::MessagesProduct) = error("mirroredlogmean() is not defined for $(product). MessagesProduct structure has to be approximated and cannot be used in inference procedure.")
loggammamean(product::MessagesProduct)    = error("loggammamean() is not defined for $(product). MessagesProduct structure has to be approximated and cannot be used in inference procedure.")


## Utility functions

as_message(message::Message)             = message
as_message(vmessage::VariationalMessage) = materialize!(vmessage)

## Operators

# TODO
reduce_messages(messages) = mapreduce(as_message, (left, right) -> multiply_messages(ProdPreserveParametrisation(), left, right), messages)

## Message Mapping structure
## Explanation: Julia cannot fully infer type of the lambda callback function in activate! method in node.jl file
## We create a lambda-like callable structure to improve type inference and make it more stable
## However it is not fully inferrable due to dynamic tags and variable constraints, but still better than just a raw lambda callback

struct MessageMapping{F, E, T, C, N, M, A, R}
    fform           :: E
    vtag            :: T
    vconstraint     :: C
    msgs_names      :: N
    marginals_names :: M
    meta            :: A
    factornode      :: R
end

message_mapping_fform(m::MessageMapping{F}) where F = F
message_mapping_fform(m::MessageMapping{F}) where F <: Function = m.fform

function MessageMapping(::Type{F}, vtag::T, vconstraint::C, msgs_names::N, marginals_names::M, meta::A, factornode::R) where { F, T, C, N, M, A, R }
    return MessageMapping{F, Nothing, T, C, N, M, A, R}(nothing, vtag, vconstraint, msgs_names, marginals_names, meta, factornode)
end

function MessageMapping(fn::E, vtag::T, vconstraint::C, msgs_names::N, marginals_names::M, meta::A, factornode::R) where { E <: Function, T, C, N, M, A, R} 
    return MessageMapping{E, E, T, C, N, M, A, R}(fn, vtag, vconstraint, msgs_names, marginals_names, meta, factornode)
end

function (mapping::MessageMapping)(dependencies)
    messages  = dependencies[1]
    marginals = dependencies[2]

    # Message is clamped if all of the inputs are clamped
    is_message_clamped = __check_all(is_clamped, messages) && __check_all(is_clamped, marginals)

    # Message is initial if it is not clamped and all of the inputs are either clamped or initial
    is_message_initial = !is_message_clamped && (__check_all(m -> is_clamped(m) || is_initial(m), messages) && __check_all(m -> is_clamped(m) || is_initial(m), marginals))

    message = rule(
        message_mapping_fform(mapping), 
        mapping.vtag, 
        mapping.vconstraint, 
        mapping.msgs_names, 
        messages, 
        mapping.marginals_names, 
        marginals, 
        mapping.meta, 
        mapping.factornode
    )

    return Message(message, is_message_clamped, is_message_initial)
end

Base.map(::Type{T}, mapping::M) where { T, M <: MessageMapping } = Rocket.MapOperator{T, M}(mapping)


