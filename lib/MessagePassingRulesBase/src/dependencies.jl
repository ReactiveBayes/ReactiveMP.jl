"""
    DependencySelector

How a dependency selects from its interface: the interface itself ([`SingleInterface`](@ref)), or,
for a group, all members ([`AllGroupMembers`](@ref)), the member aligned with the target
([`AlignedGroupMember`](@ref)), all but that one ([`AllGroupMembersButSelf`](@ref)), or a custom function
([`select_group_members`](@ref)).
"""
abstract type DependencySelector end

"""`m[:μ]`, or a cluster `q[:y, :x]`."""
struct SingleInterface <: DependencySelector end
"""`m[:in...]`: every member of the group."""
struct AllGroupMembers <: DependencySelector end
"""`m[:in][k]`: the member with the target's index."""
struct AlignedGroupMember <: DependencySelector end
"""`m[:in][!k]`: every member except the one with the target's index."""
struct AllGroupMembersButSelf <: DependencySelector end

"""
    CustomGroupSelector

A user selector made by [`select_group_members`](@ref).
"""
struct CustomGroupSelector{F} <: DependencySelector
    f::F
    arity::Int
end

"""
    select_group_members(f; arity)

A custom group selector: `f(k)` returns the tuple of member indices for target index `k`,
always of length `arity`. The arity must be static; a selector that returns another length
is an error when it is resolved.
"""
select_group_members(f; arity::Integer) = CustomGroupSelector(f, Int(arity))

"""
    selected_indices(selector, k, n)

The member indices `selector` picks from a group of `n` for target index `k`. A selection
of no members is an empty tuple; an engine must treat it as satisfied, never wait on it.

```jldoctest
julia> using MessagePassingRulesBase: selected_indices, AllGroupMembersButSelf, select_group_members

julia> selected_indices(AllGroupMembersButSelf(), 2, 4)
(1, 3, 4)

julia> selected_indices(AllGroupMembersButSelf(), 1, 1)
()

julia> selected_indices(select_group_members(k -> (mod1(k - 1, 3),); arity = 1), 1, 3)
(3,)
```
"""
selected_indices(::AllGroupMembers, k, n) = ntuple(identity, n)
selected_indices(::AlignedGroupMember, k, n) = (k,)
selected_indices(::AllGroupMembersButSelf, k, n) = ntuple(i -> i < k ? i : i + 1, n - 1)
function selected_indices(selector::CustomGroupSelector, k, n)
    selected = Tuple(selector.f(k))
    length(selected) == selector.arity ||
        throw(ArgumentError("custom selector declared arity $(selector.arity) but returned $selected for index $k"))
    return selected
end

"""
    selection_arity(selector, n)

How many members `selector` picks from a group of `n`.
"""
selection_arity(::AllGroupMembers, n) = n
selection_arity(::AlignedGroupMember, n) = 1
selection_arity(::AllGroupMembersButSelf, n) = n - 1
selection_arity(selector::CustomGroupSelector, n) = selector.arity

"""
    Dependency

One input a target depends on: a message (`:m`) or marginal (`:q`) of an interface, or of
a cluster when `key` is a tuple, selected by a [`DependencySelector`](@ref).
"""
struct Dependency
    container::Symbol
    key::Union{Symbol, Tuple{Vararg{Symbol}}}
    selector::DependencySelector
end

"""
    TargetDependencies

The inputs one target consumes, `edge` being indexed for a group member. When `default` is
true, the target was declared with `default`: it consumes the engine's default scheme's inputs,
which follow the factorisation, and `inputs` are added to them.
"""
struct TargetDependencies
    edge::Symbol
    indexed::Bool
    inputs::Tuple{Vararg{Dependency}}
    default::Bool
end

TargetDependencies(edge::Symbol, indexed::Bool, inputs::Tuple{Vararg{Dependency}}) = TargetDependencies(edge, indexed, inputs, false)

"""
    DependenciesSpec

What a node consumes under one algorithm, per target, and optionally the partition free
energy is computed over. The two are separate: a rule may consume a marginal that is not a
block of the partition, and that marginal is never scored. When `partition` is `nothing`
the engine derives it from the factorisation.
"""
struct DependenciesSpec
    node::Any
    algorithm::Type
    targets::Tuple{Vararg{TargetDependencies}}
    partition::Union{Nothing, Tuple{Vararg{Tuple{Vararg{Symbol}}}}}
end

"""
    dependencies_spec(node, algorithm)

The [`DependenciesSpec`](@ref) for `node` under `algorithm`, or `nothing` when none is
declared and the engine's default scheme applies. A [`DefaultAlgorithmExtension`](@ref)
that declares none gets the default algorithm's.
"""
dependencies_spec(node, algorithm) =
    algorithm isa DefaultAlgorithmExtension ? dependencies_spec(node, DefaultAlgorithm()) : nothing

"""
    target_dependencies(declaration, target)

The inputs `target` consumes under `declaration`, or `nothing` if it declares none for it.
"""
function target_dependencies(declaration::DependenciesSpec, target)
    entry = target_entry(declaration, target)
    return entry === nothing ? nothing : entry.inputs
end

"""
    extends_default_scheme(declaration, target)

Whether `target` was declared with `default`, so that it consumes the engine's default scheme's
inputs plus those [`target_dependencies`](@ref) lists. False for a target the declaration does
not list.
"""
function extends_default_scheme(declaration::DependenciesSpec, target)
    entry = target_entry(declaration, target)
    return entry !== nothing && entry.default
end

function target_entry(declaration::DependenciesSpec, target)
    indexed = target isa IndexedTarget
    for entry in declaration.targets
        entry.edge === target_edge(target) && entry.indexed === indexed && return entry
    end
    return nothing
end

"""
    free_energy_partition(spec)
"""
free_energy_partition(spec::DependenciesSpec) = spec.partition

function validate_dependencies(spec::NodeSpec, declaration::DependenciesSpec)
    node = spec.node
    names = map(i -> i.name, spec.interfaces)
    isgroup(name) = any(i -> i.name === name && i.group, spec.interfaces)
    function known(name)
        name in names || throw(ArgumentError("$node has no interface `$name`; its interfaces are $names"))
        return name
    end
    seen_targets = Set{Tuple{Symbol, Bool}}()
    for entry in declaration.targets
        known(entry.edge)
        if entry.indexed && !isgroup(entry.edge)
            throw(ArgumentError("`$(entry.edge)` is not a group of $node; its target is written `:$(entry.edge)`"))
        elseif !entry.indexed && isgroup(entry.edge)
            throw(ArgumentError("`$(entry.edge)` is a group of $node; its targets are written `(:$(entry.edge), k)`"))
        end
        (entry.edge, entry.indexed) in seen_targets &&
            throw(ArgumentError("the target `$(entry.edge)` of $node is declared twice"))
        push!(seen_targets, (entry.edge, entry.indexed))
        seen_inputs = Set()
        for input in entry.inputs
            (input.container, input.key) in seen_inputs &&
                throw(ArgumentError("`$(input.container)[$(input.key)]` is listed twice for target `$(entry.edge)`"))
            push!(seen_inputs, (input.container, input.key))
            # Beside the default scheme's inputs, only a single interface's: its message or its
            # marginal, which the engine places among them in interface order.
            entry.default && (input.key isa Tuple || !(input.selector isa SingleInterface) || isgroup(input.key)) &&
                throw(ArgumentError("the target `$(entry.edge)` of $node extends the default scheme with a single interface's message or marginal, `m[:x]` or `q[:x]`; got `$(dependency_label(input))`"))
            if input.key isa Tuple
                foreach(known, input.key)
                length(input.key) == 1 && !isgroup(only(input.key)) &&
                    throw(ArgumentError("`q[($(repr(only(input.key))),)]` is a one-member cluster of a single interface, which is its marginal; write `q[$(repr(only(input.key)))]`"))
                positions = map(member -> findfirst(==(member), names), input.key)
                issorted(positions) ||
                    throw(ArgumentError("the cluster `q[$(join(repr.(input.key), ", "))]` must list its members in interface order, $(Tuple(names[sort(collect(positions))]))"))
            elseif input.selector isa SingleInterface
                known(input.key)
                isgroup(input.key) &&
                    throw(ArgumentError("`$(input.key)` is a group; write `$(input.container)[:$(input.key)...]`, `$(input.container)[:$(input.key)][k]` or `$(input.container)[:$(input.key)][!k]` to select from group `$(input.key)`"))
            else
                known(input.key)
                isgroup(input.key) || throw(ArgumentError("`$(input.key)` is not a group of $node, so it has no members to select"))
                (input.selector isa AllGroupMembers || entry.indexed) ||
                    throw(ArgumentError("selecting members of `$(input.key)` by the target's index needs an indexed target, like `(:$(entry.edge), k)`"))
            end
        end
    end
    blocks = declaration.partition
    if blocks !== nothing
        covered = Symbol[]
        for block in blocks, name in block
            known(name)
            name in covered && throw(ArgumentError("`$name` appears in more than one partition block"))
            push!(covered, name)
        end
        for name in names
            name in covered || throw(ArgumentError("the partition of $node does not cover `$name`"))
        end
    end
    return declaration
end

# Macro side: `lhs => rhs` pairs in the `args` vocabulary.

function dependency_declaration_expr(name, node, algorithm_type, depsexpr, partitionexpr)
    (depsexpr isa Expr && depsexpr.head === :vect) ||
        error("@$name: `dependencies` must be a vector of `target => (inputs...)` pairs")
    targets = map(item -> parse_dependency_pair(name, item), depsexpr.args)
    blocks = partitionexpr === nothing ? nothing : parse_partition(name, partitionexpr)
    return :($DependenciesSpec($node, $algorithm_type, ($(targets...),), $blocks))
end

function parse_dependency_pair(name, item)
    (item isa Expr && item.head === :call && length(item.args) == 3 && item.args[1] === :(=>)) ||
        error("@$name: each dependency is `target => (inputs...)`, got `$item`")
    lhs, rhs = item.args[2], item.args[3]
    edge, index = if quoted_symbol(lhs) !== nothing
        quoted_symbol(lhs), nothing
    elseif lhs isa Expr && lhs.head === :tuple && length(lhs.args) == 2 && quoted_symbol(lhs.args[1]) !== nothing && lhs.args[2] isa Symbol
        quoted_symbol(lhs.args[1]), lhs.args[2]
    else
        error("@$name: a dependency target is `:out` or `(:m, k)`, got `$lhs`")
    end
    entries = rhs isa Expr && rhs.head === :tuple ? rhs.args : [rhs]
    # `default` stands for the default scheme's inputs, which the listed ones extend.
    defaults = count(==(:default), entries)
    defaults > 1 && error("@$name: `default` twice for the target `$lhs`")
    inputs = map(entry -> parse_dependency_input(name, entry, index), filter(!=(:default), entries))
    return :($TargetDependencies($(QuoteNode(edge)), $(index !== nothing), ($(inputs...),), $(defaults == 1)))
end

function parse_dependency_input(name, entry, index)
    (entry isa Expr && entry.head === :ref) ||
        error("@$name: a dependency input is `m[...]` or `q[...]`, got `$entry`")
    inner, selectors = entry.args[1], entry.args[2:end]
    if inner isa Expr && inner.head === :ref && length(selectors) == 1
        # `m[:p][k]`, `m[:p][!k]`, `m[:p][select(...)]`
        container, key = inner.args[1], quoted_symbol(get(inner.args, 2, nothing))
        (container in (:m, :q) && key !== nothing && length(inner.args) == 2) ||
            error("@$name: a group member is selected as `m[:p][k]`, got `$entry`")
        selector = selectors[1]
        selexpr = if index !== nothing && selector === index
            :($AlignedGroupMember())
        elseif index !== nothing && selector isa Expr && selector.head === :call && selector.args == [:!, index]
            :($AllGroupMembersButSelf())
        elseif selector isa Expr && selector.head === :call && selector.args[1] === :select_group_members
            Expr(:call, select_group_members, selector.args[2:end]...)
        elseif selector isa Symbol || (selector isa Expr && selector.head === :call && selector.args[1] === :!)
            index === nothing ?
                :($throw($ArgumentError($("selecting members of `$key` by the target's index needs an indexed target, like `(:p, k)`")))) :
                error("@$name: `$(selector isa Symbol ? selector : selector.args[2])` is not the index of this target; write `$(container)[:$key][$index]` or `$(container)[:$key][!$index]`")
        else
            error("@$name: a group member selector is `k`, `!k` or `select_group_members(f; arity)`, got `$selector`")
        end
        return :($Dependency($(QuoteNode(container)), $(QuoteNode(key)), $selexpr))
    end
    container = inner
    container in (:m, :q) || error("@$name: a dependency input is `m[...]` or `q[...]`, got `$entry`")
    if length(selectors) == 1 && selectors[1] isa Expr && selectors[1].head === :tuple
        selectors = selectors[1].args
        isempty(selectors) && error("@$name: `$entry` names no interface")
    elseif length(selectors) == 1
        key = selectors[1]
        if key isa Expr && key.head === :... && length(key.args) == 1 && quoted_symbol(key.args[1]) !== nothing
            return :($Dependency($(QuoteNode(container)), $(QuoteNode(quoted_symbol(key.args[1]))), $AllGroupMembers()))
        end
        symbol = quoted_symbol(key)
        symbol === nothing && error("@$name: an interface is a symbol like `:μ`, got `$key`")
        return :($Dependency($(QuoteNode(container)), $(QuoteNode(symbol)), $SingleInterface()))
    end
    container === :q || error("@$name: only marginals have clusters; use `q[...]`, got `$entry`")
    members = map(selectors) do key
        symbol = quoted_symbol(key)
        symbol === nothing && error("@$name: a cluster member is a symbol like `:y`, got `$key`")
        symbol
    end
    return :($Dependency(:q, $(Tuple(members)), $SingleInterface()))
end

function parse_partition(name, ex)
    (ex isa Expr && ex.head === :vect) ||
        error("@$name: `free_energy_partition` is a vector of clusters like `[(:out, :μ), (:τ,)]`")
    return Tuple(
        map(ex.args) do block
            entries = block isa Expr && block.head === :tuple ? block.args : [block]
            Tuple(
                map(entries) do entry
                    symbol = quoted_symbol(entry)
                    symbol === nothing && error("@$name: a partition block lists interface symbols, got `$entry`")
                    symbol
                end
            )
        end
    )
end

const DEPENDENCY_KEYWORDS = (:node, :algorithm, :dependencies, :free_energy_partition)

"""
    @define_dependencies(node = ..., algorithm = ..., dependencies = [...], free_energy_partition = [...])

Declare what `node`'s rules consume under `algorithm`. Each entry is `target => (inputs...)`
in the vocabulary of a rule's `args`: `m[:μ]`, `q[:μ]`, a cluster `q[(:y, :x)]` or
`q[:y, :x]` (members in interface order; `q[(:in,)]` is the joint over the group `in`), and for a group `m[:in...]` (all members), `m[:in][k]` (the target's own
index), `m[:in][!k]` (all but it) or `m[:in][select_group_members(f; arity)]`. A target with no inputs
is written `target => ()`.

`default` among a target's inputs stands for the engine's default scheme, whose inputs follow
the factorisation: `:a => (default, q[:a])` is the default scheme's inputs plus `q(a)`, and
`:y => (default,)` the default scheme alone. The inputs beside it are a single interface's, a
message or a marginal, and are consumed without being scored.

`free_energy_partition`, optional, lists the clusters free energy is computed over, covering every
interface once. What a rule consumes need not be a block of it.

A node's own `dependencies` keyword declares the same for its default algorithm.
"""
macro define_dependencies(args...)
    keywords = parse_keywords("define_dependencies", args, DEPENDENCY_KEYWORDS, (:node, :algorithm, :dependencies))
    node = keywords[:node]
    algorithm = :($algorithm_dispatch_type($(keywords[:algorithm])))
    declaration = dependency_declaration_expr("define_dependencies", node, algorithm, keywords[:dependencies], get(keywords, :free_energy_partition, nothing))
    base = MessagePassingRulesBase
    decl, dispatch = gensym(:dependencies), gensym(:dispatch)
    return esc(
        quote
            $base.@define_registry
            const $dispatch = $node_dispatch_type($node)
            const $decl = $validate_dependencies($nodespec($node), $declaration)
            $base.dependencies_spec(::$dispatch, ::$(declaration.args[3])) = $decl
            $register!($REGISTRY_NAME, $decl)
            nothing
        end
    )
end
