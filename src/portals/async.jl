export AsyncPortal, apply

struct AsyncPortal <: AbstractPortal end

apply(::AsyncPortal, factornode, tag, stream) = stream |> async()