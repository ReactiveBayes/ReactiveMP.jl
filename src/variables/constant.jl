export constvar, ConstVariable

mutable struct ConstVariable <: AbstractVariable
    marginal   :: MarginalObservable
    messageout :: MessageObservable
    nconnected :: Int
end

function ConstVariable(constant)
    marginal = MarginalObservable()
    connect!(marginal, of(Marginal(PointMass(constant), true, false, nothing)))
    messageout = MessageObservable(AbstractMessage)
    connect!(messageout, of(Message(PointMass(constant), true, false, nothing)))
    return ConstVariable(marginal, messageout, 0)
end

constvar(constant) = ConstVariable(constant)

degree(constvar::ConstVariable) = constvar.nconnected

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