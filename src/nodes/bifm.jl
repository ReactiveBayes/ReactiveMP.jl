
struct BIFM end

@node BIFM Deterministic [ output, input, zprev, znext ]

mutable struct BIFMMeta
    A :: Array{Real, 2}
    B :: Array{Real, 2}
    C :: Array{Real, 2}
    H :: Union{Array{Real, 2}, Nothing}
    ξztilde :: Union{Array{Real, 1}, Nothing}
    Wz :: Union{Array{Real, 2}, Nothing}
end

function BIFMMeta(A::Array{Real, 2}, B::Array{Real, 2}, C::Array{Real, 2})
    # todo: check dimensions A, B, C
    # A: n x n, B: n x m, C: d x n.
    return BIFMMeta(A, B, C, nothing, nothing, nothing)
end

getA(meta::BIFMMeta)                = meta.A
getB(meta::BIFMMeta)                = meta.B
getC(meta::BIFMMeta)                = meta.C
getH(meta::BIFMMeta)                = meta.H
getξztilde(meta::BIFMMeta)          = meta.ξztilde
getWz(meta::BIFMMeta)               = meta.Wz

function setH!(meta::BIFMMeta, H)
    meta.H = H
end

function setξztilde!(meta::BIFMMeta, ξztilde)
    meta.ξztilde = ξztilde
end

function setWz!(meta::BIFMMeta, Wz)
    meta.Wz = Wz
end

default_meta(::Type{ BIFM }) = error("BIFM node requires meta flag explicitly specified")
