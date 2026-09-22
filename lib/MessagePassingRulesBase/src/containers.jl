# Keys are held in canonical sorted order, so the dispatch signature a macro generates and
# the container an engine builds agree without either knowing the other's spelling. The
# sort runs in `@generated` code over type-level keys: at compile time, never at run time.

canonical_order(keys) = Tuple(sort!(collect(keys)))

@generated function canonical_keys(nt::NamedTuple{N}) where {N}
    sorted = canonical_order(N)
    return :(NamedTuple{$sorted}(($((:(getfield(nt, $(QuoteNode(k)))) for k in sorted)...),)))
end

"""
    Messages

The inbound messages a rule receives, reached as `args.m[:name]`. A variadic group is an
ordinary tuple under its group name: `args.m[:inputs][k]`.
"""
struct Messages{N, T <: Tuple}
    values::NamedTuple{N, T}
    Messages{N, T}(values::NamedTuple{N, T}) where {N, T} = new{N, T}(values)
end

function Messages(values::NamedTuple)
    sorted = canonical_keys(values)
    return Messages{keys(sorted), typeof(Tuple(sorted))}(sorted)
end

Base.keys(::Messages{N}) where {N} = N

@inline function Base.getindex(m::Messages{N}, key::Symbol) where {N}
    key in N || throw(KeyError(key))
    return getfield(m.values, key)
end

"""
    Marginals

The marginals a rule receives. Single interfaces and groups are reached like messages,
`args.q[:name]` and `args.q[:p][k]`. A structural cluster is reached by its members,
`args.q[:y, :x]`, in interface-declaration order.

A cluster's key is the tuple of its member names, carried in the type (`J`). It is never
turned into a symbol, so no name is ever derived and none can collide.

```jldoctest
julia> using MessagePassingRulesBase: Marginals

julia> q = Marginals((τ = 2.0, y_x = 1.0), Val(((:y, :x),)), (0.5,));

julia> q[:τ], q[:y, :x], q[:y_x]
(2.0, 0.5, 1.0)
```
"""
struct Marginals{N, T <: Tuple, J, JT <: Tuple}
    singles::NamedTuple{N, T}
    joints::JT
    Marginals{N, T, J, JT}(singles::NamedTuple{N, T}, joints::JT) where {N, T, J, JT} =
        new{N, T, J, JT}(singles, joints)
end

Marginals(singles::NamedTuple = NamedTuple()) = Marginals(singles, Val(()), ())

@generated function Marginals(singles::NamedTuple, ::Val{J}, joints::Tuple) where {J}
    length(J) == length(joints.parameters) ||
        return :(throw(ArgumentError("got $(length(J)) cluster keys but $(length(joints)) clusters")))
    order = sortperm(collect(J))
    sortedkeys = Tuple(J[order])
    sortedvalues = Expr(:tuple, (:(joints[$i]) for i in order)...)
    return quote
        sorted = canonical_keys(singles)
        values = $sortedvalues
        Marginals{keys(sorted), typeof(Tuple(sorted)), $sortedkeys, typeof(values)}(sorted, values)
    end
end

Base.keys(::Marginals{N}) where {N} = N

@inline function Base.getindex(q::Marginals{N}, key::Symbol) where {N}
    key in N || throw(KeyError(key))
    return getfield(q.singles, key)
end

@inline Base.getindex(q::Marginals, a::Symbol, b::Symbol) = q[Val((a, b))]
@inline Base.getindex(q::Marginals, a::Symbol, b::Symbol, c::Symbol) = q[Val((a, b, c))]
@inline Base.getindex(q::Marginals, a::Symbol, b::Symbol, c::Symbol, rest::Symbol...) =
    q[Val((a, b, c, rest...))]

@generated function Base.getindex(q::Marginals{N, T, J}, ::Val{K}) where {N, T, J, K}
    position = findfirst(==(K), J)
    position === nothing && return :(throw(KeyError($(QuoteNode(K)))))
    return :(getfield(q.joints, $position))
end

"""
    RuleArgs(; m = NamedTuple(), q = NamedTuple())

The arguments object a rule body receives: `args.m` holds the inbound messages and
`args.q` the marginals.
"""
struct RuleArgs{M <: Messages, Q <: Marginals}
    m::M
    q::Q
end

RuleArgs(; m = NamedTuple(), q = NamedTuple()) = RuleArgs(as_messages(m), as_marginals(q))

as_messages(m::Messages) = m
as_messages(m::NamedTuple) = Messages(m)
as_marginals(q::Marginals) = q
as_marginals(q::NamedTuple) = Marginals(q)
