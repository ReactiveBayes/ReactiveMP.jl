export StreamTransformer
export DiscontinueStreamTransformer, AsyncStreamTransformer
export DefaultMessageOutTransformer

import Base: +

abstract type StreamTransformer end

struct DiscontinueStreamTransformer <: StreamTransformer end

apply(::DiscontinueStreamTransformer, stream) = stream |> discontinue()

struct AsyncStreamTransformer <: StreamTransformer end

apply(::AsyncStreamTransformer, stream) = stream |> async(0)

struct CompositeStreamTransformer{T} <: StreamTransformer
    transformers :: T
end

apply(composite::CompositeStreamTransformer, stream) = reduce((stream, transformer) -> apply(transformer, stream), composite.transformers, init = stream)

Base.:+(left::StreamTransformer, right::StreamTransformer)                   = CompositeStreamTransformer((left, right))
Base.:+(left::StreamTransformer, right::CompositeStreamTransformer)          = CompositeStreamTransformer((left, right.transformers...))
Base.:+(left::CompositeStreamTransformer, right::StreamTransformer)          = CompositeStreamTransformer((left.transformers..., right))
Base.:+(left::CompositeStreamTransformer, right::CompositeStreamTransformer) = CompositeStreamTransformer((left.transformers..., right.transformers...))

DefaultMessageOutTransformer() = DiscontinueStreamTransformer()
