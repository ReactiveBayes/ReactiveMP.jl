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

function create_messagein!(constvar::ConstVariable)
    constvar.nconnected += 1
    return constvar.messageout, 1
end

function messagein(::ConstVariable, ::Int)
    error("ConstVariable does not save inbound messages.")
end

function messageout(constvar::ConstVariable, ::Int)
    return constvar.messageout
end

_getmarginal(constvar::ConstVariable)      = constvar.marginal
_setmarginal!(::ConstVariable, observable) = error("It is not possible to set a marginal stream for `ConstVariable`")
_makemarginal(::ConstVariable)             = error("It is not possible to make marginal stream for `ConstVariable`")
