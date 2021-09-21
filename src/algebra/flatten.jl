# type stable flatten function for tuples
flatten() = ()
flatten(a::Tuple) = Tuple(a)
flatten(a) = (a,)
flatten(a::Tuple, b...) = tuple(flatten(a...)..., flatten(b...)...)
flatten(a, b...) = tuple(a, flatten(b...)...)
flatten_tuple(x::Tuple) = flatten(x...)