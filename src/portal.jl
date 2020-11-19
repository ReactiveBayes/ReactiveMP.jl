export StreamPortal
export EmptyStreamPortal, DiscontinueStreamPortal, AsyncStreamPortal, LoggerStreamPortal, MapStreamPortal
export DefaultMessageOutPortal

import Base: +

function message_out_portal end

DefaultMessageOutPortal() = DiscontinueStreamPortal()

## Abstract Stream Portal

abstract type AbstractStreamPortal end

## Empty portal

struct EmptyStreamPortal <: AbstractStreamPortal end

apply(::EmptyStreamPortal, factornode, tag, stream) = stream

## Discontinue portal

struct DiscontinueStreamPortal <: AbstractStreamPortal end

apply(::DiscontinueStreamPortal, factornode, tag, stream) = stream |> discontinue()

## Async portal

struct AsyncStreamPortal <: AbstractStreamPortal end

apply(::AsyncStreamPortal, factornode, tag, stream) = stream |> async()

## Logger portal

struct LoggerStreamPortal <: AbstractStreamPortal end

apply(::LoggerStreamPortal, factornode, tag, stream) = stream |> tap((v) -> println("[Log][$(functionalform(factornode))][$(tag)]: $v"))

## Map portal

struct MapStreamPortal{F} <: AbstractStreamPortal 
    mappingFn :: F
end

apply(portal::MapStreamPortal, factornode, tag, stream) = stream |> map((v) -> portal.mappingFn(factornode, tag, v))

## Composite portal

struct CompositeStreamPortal{T} <: AbstractStreamPortal
    portals :: T
end

apply(composite::CompositeStreamPortal, factornode, tag, stream) = reduce((stream, portal) -> apply(portal, factornode, tag, stream), composite.portals, init = stream)

Base.:+(left::AbstractStreamPortal,  right::AbstractStreamPortal)  = CompositeStreamPortal((left, right))
Base.:+(left::AbstractStreamPortal,  right::CompositeStreamPortal) = CompositeStreamPortal((left, right.transformers...))
Base.:+(left::CompositeStreamPortal, right::AbstractStreamPortal)  = CompositeStreamPortal((left.transformers..., right))
Base.:+(left::CompositeStreamPortal, right::CompositeStreamPortal) = CompositeStreamPortal((left.transformers..., right.transformers...))
