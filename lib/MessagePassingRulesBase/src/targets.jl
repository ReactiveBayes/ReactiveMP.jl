# An element of a cluster's key: an interface, `:out`, or one member of a group, `(:T, 1)`.
const ClusterMember = Union{Symbol, Tuple{Symbol, Int}}

# Joint keys sort by their elements' names and then members' indices, which for keys of names
# is the order of the names themselves. Both a rule's dispatch signature and the marginals an
# engine builds use it.
cluster_sort_key(key) = map(element -> element isa Symbol ? (String(element), 0) : (String(first(element)), last(element)), key)

"""
    Target{E}

The outbound edge `E` of a rule, written `target = :out`.
"""
struct Target{E} end

Target(edge::Symbol) = Target{edge}()

"""
    IndexedTarget{E}(index)

Member `index` of the interface group `E`, written `target = (:m, k)`. The index is a
field rather than a type parameter, so every position of a group shares one rule.
"""
struct IndexedTarget{E}
    index::Int
end

IndexedTarget(edge::Symbol, index::Integer) = IndexedTarget{edge}(index)

"""
    target_edge(target)

The interface name a target points at.
"""
target_edge(::Target{E}) where {E} = E
target_edge(::IndexedTarget{E}) where {E} = E

"""
    target_index(target::IndexedTarget)

The group position an indexed target points at.
"""
target_index(target::IndexedTarget) = target.index

"""
    ClusterTarget{K}

The structural cluster a marginal rule computes, written `target = (:y, :x)` with the
members in interface-declaration order. A member of a group is written `(:T, 1)`, as in
`(:out, (:T, 1))`, the joint of `out` and the group `T`'s first member.
"""
struct ClusterTarget{K} end

ClusterTarget(members::Tuple{Vararg{ClusterMember}}) = ClusterTarget{members}()

"""
    cluster_members(target::ClusterTarget)

The interface names of a cluster, in declaration order.
"""
cluster_members(::ClusterTarget{K}) where {K} = K
