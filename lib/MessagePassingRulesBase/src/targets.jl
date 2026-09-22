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
    edge(target)

The interface name a target points at.
"""
edge(::Target{E}) where {E} = E
edge(::IndexedTarget{E}) where {E} = E

"""
    index(target::IndexedTarget)

The group position an indexed target points at.
"""
index(target::IndexedTarget) = target.index
