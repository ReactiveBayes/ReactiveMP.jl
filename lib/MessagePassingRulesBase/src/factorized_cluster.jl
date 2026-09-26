"""
    FactorizedCluster(block => distribution, ...)

The result of a marginal rule whose cluster factorises into independent blocks:
`FactorizedCluster((:out, :μ) => q_outμ, (:v,) => q_v)` for `q(out, μ, v) = q(out, μ) q(v)`.

It is a BayesBase `FactorizedJoint` of the blocks, labelled with the members each block covers.
The joint is the distribution, and BayesBase supplies its `entropy` and float-type conversion.
The labels are what an engine needs to hand each block on and to score it. Each label is the
tuple of the block's members, in the cluster's order; a member of a group is written as in a
cluster's key, `(:T, 1)`. The labels are carried in the type, so a block is read as
`fc[(:out, :μ)]` with no lookup at run time, and no names are joined together, so interface
names may contain underscores. `pairs(fc)` gives `block => distribution` pairs, and
`BayesBase.components(fc)` the distributions.

# Throws
- `ArgumentError` when given no block;
- `KeyError` when indexed by a block it does not have.

# Examples

```jldoctest
julia> fc = FactorizedCluster((:out, :μ) => 1.0, (:v,) => 2.0);

julia> fc[(:v,)]
2.0

julia> MessagePassingRulesBase.cluster_blocks(fc)
((:out, :μ), (:v,))
```

See also [`cluster_blocks`](@ref), [`check_factorized_cluster`](@ref).
"""
struct FactorizedCluster{K, J <: FactorizedJoint}
    joint::J
    FactorizedCluster{K, J}(joint::J) where {K, J} = new{K, J}(joint)
end

# The labels reach the type through `Val`, which constant propagation resolves when a rule
# body writes them literally; `gate:factorized-cluster` measures it.
@inline FactorizedCluster(blocks::Pair{<:Tuple{Vararg{ClusterMember}}}...) = FactorizedCluster(Val(map(first, blocks)), map(last, blocks))
@inline FactorizedCluster(::Val{K}, components::Tuple) where {K} = FactorizedCluster(Val(K), FactorizedJoint(components))
@inline FactorizedCluster(::Val{K}, joint::FactorizedJoint) where {K} = FactorizedCluster{K, typeof(joint)}(joint)

FactorizedCluster() = throw(ArgumentError("a FactorizedCluster needs at least one block"))

"""
    cluster_blocks(fc::FactorizedCluster) -> Tuple

The labels of the blocks of `fc`, in order: one tuple of members per block.
"""
cluster_blocks(::FactorizedCluster{K}) where {K} = K

BayesBase.components(fc::FactorizedCluster) = BayesBase.components(fc.joint)

@inline Base.getindex(fc::FactorizedCluster, block::Tuple{Vararg{ClusterMember}}) = fc[Val(block)]

@generated function Base.getindex(fc::FactorizedCluster{K}, ::Val{B}) where {K, B}
    position = findfirst(==(B), K)
    position === nothing && return :(throw(KeyError($(QuoteNode(B)))))
    return :(getfield(BayesBase.components(fc.joint), $position))
end

Base.pairs(fc::FactorizedCluster{K}) where {K} = map(Pair, K, BayesBase.components(fc))

BayesBase.entropy(fc::FactorizedCluster) = BayesBase.entropy(fc.joint)
BayesBase.paramfloattype(fc::FactorizedCluster) = BayesBase.paramfloattype(fc.joint)
BayesBase.convert_paramfloattype(::Type{T}, fc::FactorizedCluster{K}) where {T, K} =
    FactorizedCluster(Val(K), BayesBase.convert_paramfloattype(T, fc.joint))

function Base.show(io::IO, fc::FactorizedCluster)
    print(io, "FactorizedCluster(")
    join(io, (sprint(show, block; context = io) * " => " * sprint(show, component; context = io) for (block, component) in pairs(fc)), ", ")
    return print(io, ")")
end

"""
    check_factorized_cluster(target::ClusterTarget, fc::FactorizedCluster) -> FactorizedCluster

Check that the blocks of `fc` partition the members of `target`, every member in exactly one
block and each block listing its members in the cluster's order, and return `fc`. For a marginal rule's
tests; the blocks may come in any order.

# Throws
`ArgumentError` naming the problem: a member outside the cluster, a member in two blocks, a
block out of order, or a member no block covers.
"""
function check_factorized_cluster(target::ClusterTarget, fc::FactorizedCluster)
    members = cluster_members(target)
    covered = Symbol[]
    for block in cluster_blocks(fc)
        for member in block
            member in members || throw(ArgumentError("`$member` is not a member of the cluster $members"))
            member in covered && throw(ArgumentError("`$member` is in more than one block of $fc"))
            push!(covered, member)
        end
        issorted(map(member -> findfirst(==(member), members), block)) ||
            throw(ArgumentError("the block $block must list its members in cluster order, $members"))
    end
    for member in members
        member in covered || throw(ArgumentError("$fc does not cover `$member` of the cluster $members"))
    end
    return fc
end
