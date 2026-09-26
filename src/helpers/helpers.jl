using Rocket

import Base: show, similar
import Base: IteratorSize, HasLength
import Base: IteratorEltype, HasEltype
import Base: eltype, length, size, sum
import Base: IndexStyle, IndexLinear, getindex

import LinearAlgebra: UniformScaling

import Rocket: similar_typeof


##

# Symbol helpers

unval(::Type{Val{S}}) where {S} = S
unval(::Val{S}) where {S} = S

##

__check_all(fn::Function, iterator) = all(fn, iterator)
__check_all(fn::Function, tuple::Tuple) = TupleTools.prod(map(fn, tuple))
__check_all(fn::Function, ::Nothing) = true

##

is_clamped_or_initial(something) =
    is_clamped(something) || is_initial(something)

##

forward_range(range::OrdinalRange)::UnitRange =
    step(range) > 0 ? (first(range):last(range)) : (last(range):first(range))
