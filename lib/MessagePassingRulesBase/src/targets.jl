# An element of a cluster's key: an interface, `:out`, or one member of a group, `(:T, 1)`.
const ClusterMember = Union{Symbol, Tuple{Symbol, Int}}

# Joint keys sort by their elements' names and then members' indices, which for keys of names
# is the order of the names themselves. Both a rule's dispatch signature and the marginals an
# engine builds use it.
cluster_sort_key(key) = map(element -> element isa Symbol ? (String(element), 0) : (String(first(element)), last(element)), key)

"""
    Target(edge::Symbol)
    Target{E}()

The target of a message rule towards the single interface `E`, written `target = :out` in a
rule's declaration. The edge is a type parameter, so a rule dispatches on it. An engine and the
`message_passing_*` calls pass one to [`find_message_rule`](@ref); a rule's body never sees it.

```jldoctest
julia> using MessagePassingRulesBase: Target, target_edge

julia> target_edge(Target(:out))
:out
```

See also [`IndexedTarget`](@ref), [`ClusterTarget`](@ref).
"""
struct Target{E} end

Target(edge::Symbol) = Target{edge}()

"""
    IndexedTarget(edge::Symbol, index::Integer)
    IndexedTarget{E}(index)

The target of a message rule towards member `index` of the interface group `E`, written
`target = (:m, k)` in a rule's declaration. The index is a field rather than a type parameter,
so every member of a group shares one rule, which reads its member's index as the name `k` it
binds.

```jldoctest
julia> using MessagePassingRulesBase: IndexedTarget, target_edge, target_index

julia> t = IndexedTarget(:m, 2); (target_edge(t), target_index(t))
(:m, 2)
```

See also [`Target`](@ref), [`ClusterTarget`](@ref).
"""
struct IndexedTarget{E}
    index::Int
end

IndexedTarget(edge::Symbol, index::Integer) = IndexedTarget{edge}(index)

"""
    target_edge(target::Union{Target, IndexedTarget}) -> Symbol

The interface a message rule's target points at: the interface's name for a [`Target`](@ref),
the group's name for an [`IndexedTarget`](@ref).
"""
target_edge(::Target{E}) where {E} = E
target_edge(::IndexedTarget{E}) where {E} = E

"""
    target_index(target::IndexedTarget) -> Int

The index of the group member an [`IndexedTarget`](@ref) points at, counted from 1.
"""
target_index(target::IndexedTarget) = target.index

"""
    ClusterTarget(members::Tuple)
    ClusterTarget{K}()

The target of a marginal rule: the structural cluster `K`, written `target = (:y, :x)` in a
rule's declaration, its members in interface-declaration order. A member of a group is written
`(:T, 1)`, as in `(:out, (:T, 1))`, the joint of `out` and the group `T`'s first member. A rule
declared with a bare name, `target = members`, is defined for every `ClusterTarget` of its node.

```jldoctest
julia> using MessagePassingRulesBase: ClusterTarget, cluster_members

julia> cluster_members(ClusterTarget((:out, (:T, 1))))
(:out, (:T, 1))
```

See also [`Target`](@ref), [`IndexedTarget`](@ref).
"""
struct ClusterTarget{K} end

ClusterTarget(members::Tuple{Vararg{ClusterMember}}) = ClusterTarget{members}()

"""
    cluster_members(target::ClusterTarget) -> Tuple

The members of a cluster, in interface-declaration order: an interface's name, or `(group, k)`
for member `k` of a group.
"""
cluster_members(::ClusterTarget{K}) where {K} = K
