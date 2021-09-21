# type stable flatten function for tuples
flatten_tuple() = ()
flatten_tuple(a) = (a,)
flatten_tuple(a::Tuple, b...) = tuple(flatten_tuple(a...)..., flatten_tuple(b...)...)
flatten_tuple(a, b...) = tuple(a, flatten_tuple(b...)...)
flatten_tuple(x::Tuple) = tuple(flatten_tuple(x...)...)