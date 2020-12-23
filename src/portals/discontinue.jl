export DiscontinuePortal, apply

struct DiscontinuePortal <: AbstractPortal end

apply(::DiscontinuePortal, factornode, tag, stream) = stream |> discontinue()