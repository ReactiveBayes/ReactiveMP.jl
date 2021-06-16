export BIFM, BIFMMeta

struct BIFM end

@node BIFM Deterministic [ output, input, zprev, znext ]

mutable struct BIFMMeta
    A :: Array{Real, 2}
    B :: Array{Real, 2}
    C :: Array{Real, 2}
    H :: Union{Array{Real, 2}, Nothing}
    ξztilde :: Union{Array{Real, 1}, Nothing}
    Wz :: Union{Array{Real, 2}, Nothing}
    μu :: Union{Array{Real, 1}, Nothing}
    Σu :: Union{Array{Real, 2}, Nothing}
end

function BIFMMeta(A::Array{Real, 2}, B::Array{Real, 2}, C::Array{Real, 2})
    # check whether the dimensionality of transition matrices makes sense
    @assert size(A,1) == size(B,1)
    @assert size(A,1) == size(C,2)

    # return default Meta data for BIFM node
    return BIFMMeta(A, B, C, nothing, nothing, nothing, nothing, nothing)
end

getA(meta::BIFMMeta)                = meta.A
getB(meta::BIFMMeta)                = meta.B
getC(meta::BIFMMeta)                = meta.C
getH(meta::BIFMMeta)                = meta.H
getξztilde(meta::BIFMMeta)          = meta.ξztilde
getWz(meta::BIFMMeta)               = meta.Wz
getμu(meta::BIFMMeta)               = meta.μu
getΣu(meta::BIFMMeta)               = meta.Σu

function setH!(meta::BIFMMeta, H)
    meta.H = H
end

function setξztilde!(meta::BIFMMeta, ξztilde)
    meta.ξztilde = ξztilde
end

function setμu!(meta::BIFMMeta, μu)
    meta.μu = μu
end

function setΣu!(meta::BIFMMeta, Σu)
    meta.Σu = Σu
end


default_meta(::Type{ BIFM }) = error("BIFM node requires meta flag explicitly specified")
