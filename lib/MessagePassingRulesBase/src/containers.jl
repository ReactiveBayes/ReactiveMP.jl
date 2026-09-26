# Keys are held in canonical sorted order, so the dispatch signature a macro generates and
# the container an engine builds agree without either knowing the other's spelling. The
# sort runs in `@generated` code over type-level keys: at compile time, never at run time.

canonical_order(keys) = Tuple(sort!(collect(keys)))

"""
    canonical_cluster_keys(keys) -> Tuple

Sort joint-cluster keys into the order a [`Marginals`](@ref) holds them: by the names of their
members, a group member's index after its group's name. A rule's dispatch signature and the
containers an engine builds both use this order, so neither needs the other's spelling.

```jldoctest
julia> using MessagePassingRulesBase: canonical_cluster_keys

julia> canonical_cluster_keys(((:y, :x), (:out, (:T, 2)), (:out, (:T, 1))))
((:out, (:T, 1)), (:out, (:T, 2)), (:y, :x))
```
"""
canonical_cluster_keys(keys) = Tuple(sort(collect(keys); by = cluster_sort_key))

@generated function canonical_keys(nt::NamedTuple{N}) where {N}
    sorted = canonical_order(N)
    return :(NamedTuple{$sorted}(($((:(getfield(nt, $(QuoteNode(k)))) for k in sorted)...),)))
end

"""
    Messages(values::NamedTuple)

The inbound messages a rule receives as `args.m`, keyed by interface name: `args.m[:μ]`. A group
is a tuple of its members in order under the group's name, `args.m[:inputs][k]`, with `nothing`
for a member the rule does not take. The keys are held sorted, so the order a caller gives them
in does not matter. `keys(m)` gives the names; indexing by a name `m` does not hold throws a
`KeyError`.

```jldoctest
julia> using MessagePassingRulesBase: Messages

julia> m = Messages((μ = 1.0, inputs = (2.0, 3.0)));

julia> m[:μ], m[:inputs][2], keys(m)
(1.0, 3.0, (:inputs, :μ))
```

See also [`Marginals`](@ref), [`RuleArgs`](@ref).
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
    Marginals(singles::NamedTuple = NamedTuple())
    Marginals(singles::NamedTuple, Val(cluster_keys), joints::Tuple)

The marginals a rule receives as `args.q`. Single interfaces and groups are reached like
messages, `args.q[:name]` and `args.q[:p][k]`. A structural cluster is reached by the tuple of
its members, in interface-declaration order: `args.q[(:y, :x)]`, or `args.q[:y, :x]` for short.
Inside a cluster a group's name stands for all its members jointly, so `args.q[(:in,)]` is the
joint over the group `in`, while `args.q[:in]` is the tuple of its members' marginals. One member
of a group is `(:in, 1)`: `args.q[:out, (:in, 1)]` is the joint of `out` and that member.

The second form takes the joints' keys, as a `Val` of a tuple of member tuples, and the joints
in the same order. A cluster's key is carried in the type and never turned into a symbol, so no
name is derived and none can collide: `q[:y_x]` below is an interface named `y_x`, not the
cluster.

# Throws
- `ArgumentError` when the numbers of cluster keys and joints differ;
- `KeyError` when indexed by a name or a cluster it does not hold.

# Examples

```jldoctest
julia> using MessagePassingRulesBase: Marginals

julia> q = Marginals((τ = 2.0, y_x = 1.0), Val(((:y, :x),)), (0.5,));

julia> q[:τ], q[:y, :x], q[(:y, :x)], q[:y_x]
(2.0, 0.5, 0.5, 1.0)
```

See also [`Messages`](@ref), [`RuleArgs`](@ref).
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
    order = sortperm(collect(J); by = cluster_sort_key)
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

@inline Base.getindex(q::Marginals, members::Tuple{Vararg{ClusterMember}}) = q[Val(members)]
@inline Base.getindex(q::Marginals, a::Symbol, b::Symbol) = q[Val((a, b))]
@inline Base.getindex(q::Marginals, a::Symbol, b::Symbol, c::Symbol) = q[Val((a, b, c))]
@inline Base.getindex(q::Marginals, a::Symbol, b::Symbol, c::Symbol, rest::Symbol...) =
    q[Val((a, b, c, rest...))]
# A joint holding members of a group, `q[:out, (:T, 1)]`.
@inline Base.getindex(q::Marginals, a::ClusterMember, b::ClusterMember, rest::ClusterMember...) = q[Val((a, b, rest...))]

@generated function Base.getindex(q::Marginals{N, T, J}, ::Val{K}) where {N, T, J, K}
    position = findfirst(==(K), J)
    position === nothing && return :(throw(KeyError($(QuoteNode(K)))))
    return :(getfield(q.joints, $position))
end

"""
    RuleArgs(; m = NamedTuple(), q = NamedTuple(), logscale = nothing)
    RuleArgs(m::Messages, q::Marginals[, logscale])

The inputs a rule's body receives as `args`, and the value rules dispatch on.

- `args.m`: the inbound messages, a [`Messages`](@ref);
- `args.q`: the marginals, a [`Marginals`](@ref);
- `args.logscale`: the log scales that arrived with the messages, a [`RuleLogScales`](@ref)
  read as `args.logscale.m[:out]`, when the caller tracks them, and `nothing` otherwise.

Rules dispatch on the keys and types of `m` and `q` alone; a rule reads `args.logscale` only
when declared with `reads_logscale = true`.

# Keywords
- `m`: the messages, a `NamedTuple` or a `Messages`. Default: none.
- `q`: the marginals of single interfaces, a `NamedTuple` or a `Marginals`. Joints need the
  `Marginals` constructor. Default: none.
- `logscale`: the messages' log scales, a `NamedTuple` keyed like `m`, or `nothing` when they are
  not tracked. Default: `nothing`.

# Examples

```jldoctest
julia> using MessagePassingRulesBase: RuleArgs

julia> args = RuleArgs(m = (μ = 1.0,), q = (v = 2.0,));

julia> args.m[:μ], args.q[:v], args.logscale
(1.0, 2.0, nothing)
```
"""
struct RuleArgs{M <: Messages, Q <: Marginals, L}
    m::M
    q::Q
    logscale::L
end

RuleArgs(m::Messages, q::Marginals) = RuleArgs(m, q, nothing)
RuleArgs(; m = NamedTuple(), q = NamedTuple(), logscale = nothing) = RuleArgs(as_messages(m), as_marginals(q), as_logscales(logscale))

as_messages(m::Messages) = m
as_messages(m::NamedTuple) = Messages(m)
as_marginals(q::Marginals) = q
as_marginals(q::NamedTuple) = Marginals(q)
as_logscales(::Nothing) = nothing
as_logscales(logscale::NamedTuple) = RuleLogScales(m = logscale)
