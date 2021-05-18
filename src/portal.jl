export AbstractPortal, EmptyPortal, CompositePortal, apply

import Base: +

## Abstract Stream Portal

abstract type AbstractPortal end

## Default portal

struct EmptyPortal <: AbstractPortal end

apply(::EmptyPortal, factornode, tag, stream) = stream

## Composite portal

struct CompositePortal{T} <: AbstractPortal
    portals :: T
end

apply(composite::CompositePortal, factornode, tag, stream) = reduce((stream, portal) -> apply(portal, factornode, tag, stream), composite.portals, init = stream)

Base.:+(left::EmptyPortal,     right::EmptyPortal)     = EmptyPortal()
Base.:+(left::EmptyPortal,     right::AbstractPortal)  = right
Base.:+(left::AbstractPortal,  right::EmptyPortal)     = left
Base.:+(left::AbstractPortal,  right::AbstractPortal)  = CompositePortal((left, right))
Base.:+(left::AbstractPortal,  right::CompositePortal) = CompositePortal((left, right.portals...))
Base.:+(left::CompositePortal, right::AbstractPortal)  = CompositePortal((left.portals..., right))
Base.:+(left::CompositePortal, right::CompositePortal) = CompositePortal((left.portals..., right.portals...))
