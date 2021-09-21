# type stable flatten function for tuples
flatten_tuple() = ()
flatten_tuple(a::Tuple) = Tuple(a)
flatten_tuple(a) = (a,)
flatten_tuple(a::Tuple, b...) = tuple(flatten_tuple(a...)..., flatten_tuple(b...)...)
flatten_tuple(a, b...) = tuple(a, flatten_tuple(b...)...)
flatten_tuple(x::Tuple) = flatten_tuple(x...)