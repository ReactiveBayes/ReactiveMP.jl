export constvar, ConstVariable

"""
    ConstVariable <: AbstractVariable

A constant of the model, fixed when it is created. Its message and its marginal are a clamped
`PointMass` of the constant, with log scale zero, wired at creation: a constant needs no
activation, and every node connected to it shares its one message stream. A node's interface on
a constant sends it no message. Create one with [`constvar`](@ref).

See also [`RandomVariable`](@ref), [`DataVariable`](@ref).
"""
mutable struct ConstVariable <: AbstractVariable
    marginal::MarginalObservable
    messageout::MessageObservable
    constant::Any
    nconnected::Int
    label::Any
end

function ConstVariable(constant; label = nothing)
    marginal = MarginalObservable()
    # A point mass observed: its log scale is zero.
    connect!(marginal, of(Marginal(PointMass(constant), true, false, AnnotationDict(), 0)))
    messageout = MessageObservable(AbstractMessage)
    connect!(messageout, of(Message(PointMass(constant), true, false, AnnotationDict(), 0)))
    return ConstVariable(marginal, messageout, constant, 0, label)
end

"""
    constvar(constant; label = nothing) -> ConstVariable

Create a [`ConstVariable`](@ref) holding `constant`, a number, an array or any other value a
`PointMass` holds. `label` names it in the callback events and in error messages.

# Examples

```jldoctest
julia> c = constvar(2.0);

julia> ReactiveMP.isconst(c), ReactiveMP.degree(c)
(true, 0)
```
"""
constvar(constant; label = nothing) = ConstVariable(constant; label = label)

degree(constvar::ConstVariable) = constvar.nconnected
getconst(constvar::ConstVariable) = constvar.constant

israndom(::ConstVariable) = false
israndom(::AbstractArray{<:ConstVariable}) = false
isdata(::ConstVariable) = false
isdata(::AbstractArray{<:ConstVariable}) = false
isconst(::ConstVariable) = true
isconst(::AbstractArray{<:ConstVariable}) = true

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

set_initial_message!(::ConstVariable, message) = throw(
    ArgumentError("a constant's message is its value, a point mass: it takes no initial message"),
)

function get_stream_of_outbound_messages(constvar::ConstVariable, ::Int)
    return constvar.messageout
end
