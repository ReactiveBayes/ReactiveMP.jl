"""
    Target{E}

The outbound edge `E` of a rule, written `towards = :out`.
"""
struct Target{E} end

Target(edge::Symbol) = Target{edge}()

"""
    IndexedTarget{E}(index)

Member `index` of the interface group `E`, written `towards = (:m, k)`. The index is a
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

The structural cluster a marginal rule computes, written `towards = (:y, :x)` with the
members in interface-declaration order.
"""
struct ClusterTarget{K} end

ClusterTarget(members::Tuple{Vararg{Symbol}}) = ClusterTarget{members}()

"""
    cluster_members(target::ClusterTarget)

The interface names of a cluster, in declaration order.
"""
cluster_members(::ClusterTarget{K}) where {K} = K
