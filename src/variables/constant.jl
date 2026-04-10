export constvar, ConstVariable

"""
    ConstVariable <: AbstractVariable

Represents a constant (clamped) variable in the factor graph. The value is fixed at creation time and
wrapped in a `PointMass` distribution. Messages and marginals from this variable are always marked as clamped.
Use [`constvar`](@ref) to create an instance.

See also: [`ReactiveMP.RandomVariable`](@ref), [`ReactiveMP.DataVariable`](@ref)
"""
mutable struct ConstVariable <: AbstractVariable
    marginal   :: MarginalObservable
    messageout :: MessageObservable
    constant   :: Any
    nconnected :: Int
    label      :: Any
end

function ConstVariable(constant; label = nothing)
    marginal = MarginalObservable()
    connect!(marginal, of(Marginal(PointMass(constant), true, false)))
    messageout = MessageObservable(AbstractMessage)
    connect!(messageout, of(Message(PointMass(constant), true, false)))
    return ConstVariable(marginal, messageout, constant, 0, label)
end

"""
    constvar(constant; label = nothing)

Creates a new [`ReactiveMP.ConstVariable`](@ref) with the given `constant` value and an optional `label` for identification.
"""
constvar(constant; label = nothing) = ConstVariable(constant; label = label)

degree(constvar::ConstVariable) = constvar.nconnected
getconst(constvar::ConstVariable) = constvar.constant

israndom(::ConstVariable)                  = false
israndom(::AbstractArray{<:ConstVariable}) = false
isdata(::ConstVariable)                    = false
isdata(::AbstractArray{<:ConstVariable})   = false
isconst(::ConstVariable)                   = true
isconst(::AbstractArray{<:ConstVariable})  = true

get_stream_of_marginals(constvar::ConstVariable) = constvar.marginal
get_stream_of_predictions(constvar::ConstVariable) = constvar.marginal

set_stream_of_marginals!(constvar::ConstVariable, stream) = error(
    "It is not possible to set a stream of marginals for a `ConstVariable`"
)
set_stream_of_predictions!(constvar::ConstVariable, stream) = error(
    "It is not possible to set a stream of predictions for a `ConstVariable`",
)

function create_new_stream_of_inbound_messages!(constvar::ConstVariable)
    constvar.nconnected += 1
    return constvar.messageout, 1
end

function get_stream_of_inbound_messages(::ConstVariable, ::Int)
    error("ConstVariable does not save inbound messages.")
end

function get_stream_of_outbound_messages(constvar::ConstVariable, ::Int)
    return constvar.messageout
end
