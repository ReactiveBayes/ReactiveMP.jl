export AbstractPortal
export EmptyPortal, DiscontinuemPortal, AsyncPortal, LoggerPortal, MapPortal
export DefaultOutboundMessagePortal

import Base: +

DefaultOutboundMessagePortal() = DiscontinuePortal()

## Abstract Stream Portal

abstract type AbstractPortal end

## Empty portal

struct EmptyPortal <: AbstractPortal end

apply(::EmptyPortal, factornode, tag, stream) = stream

## Discontinue portal

struct DiscontinuePortal <: AbstractPortal end

apply(::DiscontinuePortal, factornode, tag, stream) = stream |> discontinue()

## Async portal

struct AsyncPortal <: AbstractPortal end

apply(::AsyncPortal, factornode, tag, stream) = stream |> async()

## Logger portal

struct LoggerPortal <: AbstractPortal end

apply(::LoggerPortal, factornode, tag, stream) = stream |> tap((v) -> println("[Log][$(functionalform(factornode))][$(tag)]: $v"))

## Map portal

struct MapPortal{F} <: AbstractPortal 
    mappingFn :: F
end

apply(portal::MapPortal, factornode, tag, stream) = stream |> map((v) -> portal.mappingFn(factornode, tag, v))

## Composite portal

struct CompositePortal{T} <: AbstractPortal
    portals :: T
end

apply(composite::CompositePortal, factornode, tag, stream) = reduce((stream, portal) -> apply(portal, factornode, tag, stream), composite.portals, init = stream)

Base.:+(left::AbstractPortal,  right::AbstractPortal)  = CompositePortal((left, right))
Base.:+(left::AbstractPortal,  right::CompositePortal) = CompositePortal((left, right.transformers...))
Base.:+(left::CompositePortal, right::AbstractPortal)  = CompositePortal((left.transformers..., right))
Base.:+(left::CompositePortal, right::CompositePortal) = CompositePortal((left.transformers..., right.transformers...))
