export constvar, ConstVariable

mutable struct ConstVariable <: AbstractVariable
    marginal   :: MarginalObservable
    messageout :: MessageObservable
    constant   :: Any
    nconnected :: Int
end

function ConstVariable(constant)
    marginal = MarginalObservable()
    connect!(marginal, of(Marginal(PointMass(constant), true, false, nothing)))
    messageout = MessageObservable(AbstractMessage)
    connect!(messageout, of(Message(PointMass(constant), true, false, nothing)))
    return ConstVariable(marginal, messageout, constant, 0)
end

constvar(constant) = ConstVariable(constant)

degree(constvar::ConstVariable)          = nconnected(constvar)
name(constvar::ConstVariable)            = constvar.name
proxy_variables(constvar::ConstVariable) = nothing
collection_type(constvar::ConstVariable) = constvar.collection_type
setused!(constvar::ConstVariable)        = nothing

isproxy(::ConstVariable) = false

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
