export rule, marginalrule
export @rule, @marginalrule
export @call_rule, @call_marginalrule
export @logscale

using MacroTools
using .MacroHelpers

import Base: showerror

"""
    rule(fform, on, vconstraint, mnames, messages, qnames, marginals, meta, __node)

This function is used to compute an outbound message for a given node

# Arguments

- `fform`: Functional form of the node in form of a type of the node, e.g. `::Type{ <: NormalMeanVariance }` or `::typeof(+)`
- `on`: Outbound interface's tag for which a message has to be computed, e.g. `::Val{:out}` or `::Val{:μ}`
- `vconstraint`: Variable constraints for an outbound interface, e.g. `Marginalisation` or `MomentMatching`
- `mnames`: Ordered messages names in form of the Val type, eg. `::Val{ (:mean, :precision) }`
- `messages`: Tuple of message of the same length as `mnames` used to compute an outbound message
- `qnames`: Ordered marginal names in form of the Val type, eg. `::Val{ (:mean, :precision) }`
- `marginals`: Tuple of marginals of the same length as `qnames` used to compute an outbound message
- `meta`: Extra meta information
- `addons`: Extra addons information
- `__node`: Node reference

For all available rules, see `ReactiveMP.print_rules_table()`.

See also: [`@rule`](@ref), [`marginalrule`](@ref), [`@marginalrule`](@ref)
"""
function rule end

"""
    marginalrule(fform, on, mnames, messages, qnames, marginals, meta, __node)

This function is used to compute a local joint marginal for a given node

# Arguments

- `fform`: Functional form of the node in form of a type of the node, e.g. `::Type{ <: NormalMeanVariance }` or `::typeof(+)`
- `on`: Local joint marginal tag , e.g. `::Val{ :mean_precision }` or `::Val{ :out_mean_precision }`
- `mnames`: Ordered messages names in form of the Val type, eg. `::Val{ (:mean, :precision) }`
- `messages`: Tuple of message of the same length as `mnames` used to compute an outbound message
- `qnames`: Ordered marginal names in form of the Val type, eg. `::Val{ (:mean, :precision) }`
- `marginals`: Tuple of marginals of the same length as `qnames` used to compute an outbound message
- `meta`: Extra meta information
- `__node`: Node reference

See also: [`rule`](@ref), [`@rule`](@ref) [`@marginalrule`](@ref)
"""
function marginalrule end

# Macro code

"""
    rule_macro_parse_on_tag(expression)

Do not use this function directly. This function is private and does not belong to the public API.

This function is used to parse an `on` tag for message rules and marginal rules specification. 

```
@rule MvNormalMeanPrecision(:out, Marginalisation) (...) = begin 
                            ^^^^
                            `on` tag
    ...
end
```

or 

```
@rule NormalMixture((:m, k), Marginalisation) (...) = begin 
                    ^^^^^^^
                    `on` tag
    ...
end
```

Accepts either a quoted symbol expressions or a (name, index) tuple expression. Returns name expression, index expression and index initialisation expression.

See also: [`@rule`](@ref)
"""
function rule_macro_parse_on_tag(on)
    if @capture(on, :name_)
        # First we check on just quoted symbol expression
        # If captures index expression and index initialisation expression are `nothing`
        return :(Val{$(QuoteNode(name))}), nothing, nothing
    elseif @capture(on, (:name_, index_Symbol))
        return :(Tuple{Val{$(QuoteNode(name))}, Int}), index, :($index = on[2])
    elseif @capture(on, (:name_, k_ = index_Int))
        return :(Tuple{Val{$(QuoteNode(name))}, Int}),
        index,
        :(error("`k = ...` syntax in the edge specification is only allowed in the `@call_rule` and `@call_marginalrule` macros"))
    else
        error(
            "Error in macro. `on` specification is incorrect: $(on). Must be either a quoted symbol expression (e.g. `:out` or `:mean`) or tuple expression with quoted symbol and index identifier (e.g. `(:m, k)` or `(:w, k)`)"
        )
    end
end

"""
    rule_macro_check_fn_args(inputs; allowed_inputs, allowed_prefixes)

This function checks if all `inputs` are either in the `allowed_inputs` or have prefixes in the `allowed_prefixes`.

See also: [`@rule`](@ref)
"""
function rule_macro_check_fn_args(inputs; allowed_inputs, allowed_prefixes)
    str_allowed_inputs = map(string, allowed_inputs)
    str_allowed_prefixes = map(string, allowed_prefixes)
    foreach(inputs) do input
        str_input = string(first(input))
        if !(str_input ∈ str_allowed_inputs) && !(any(str_prefix -> startswith(str_input, str_prefix), str_allowed_prefixes))
            error(
                "Found a bad input $(input) in the macro arguments specification. It must be either in `$(str_allowed_inputs)` or start with a prefix in `$(str_allowed_prefixes)`"
            )
        end
    end
    return true
end

"""
    rule_macro_parse_fn_args(inputs; specname, prefix, proxy)

Do not use this function directly. This function is private and does not belong to the public API.

This function is used to parse an `arguments` tuple for message rules and marginal rules specification. 

```
@rule MvNormalMeanPrecision(:out, Marginalisation) (m_μ::NormalMeanPrecision, m_τ::PointMass) = begin 
                                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                                   `arguments` vector
    ...
end
```

Accepts a vector of (name, type) elements, specname, name prefix and proxy type. 
Returns parsed names without prefix, proxied types and initialisation code block.

See also: [`@rule`](@ref)
"""
function rule_macro_parse_fn_args(inputs; specname, prefix, proxy)
    # First we filter out only prefixed arguments
    finputs = filter((i) -> startswith(string(first(i)), string(prefix)), inputs)

    lprefix = length(string(prefix))

    # We extract names and types
    names = map(first, finputs)
    types = map(last, finputs)

    # Check that all arguments have proper names and not only a single prefix in it
    @assert all((n) -> length(string(n)) > lprefix, names) || error("Empty $(specname) name found in arguments")

    # Initialisation block is simply a `getdata` call from `specname` for each argument
    init_block = map(enumerate(names)) do (index, iname)
        return :($(iname) = getdata($(specname)[$(index)]))
    end

    # We return names in form of a `Type{ Val{ (:name1, :name2, ...) } }`
    # We return types in form of a `Tuple{ ProxyType{ <: type1 }, ProxyType{ <: type2 } }`
    out_names = if isempty(names)
        :Nothing
    else
        :(Val{$(Expr(:tuple, map(n -> QuoteNode(Symbol(string(n)[(lprefix + 1):end])), names)...))})
    end
    out_types = isempty(types) ? :Nothing : :(Tuple{$(map((t) -> MacroHelpers.proxy_type(proxy, t), types)...)})

    return out_names, out_types, init_block
end

"""
    call_rule_macro_parse_fn_args(inputs; specname, prefix, proxy)

Do not use this function directly. This function is private and does not belong to the public API.

This function is used to parse an `arguments` tuple for message and marginal calling rules specification. 

```
@call_rule MvNormalMeanPrecision(:out, Marginalisation) (m_μ = NormalMeanPrecision(...), m_τ = PointMass(...)) = begin 
                                                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                                        `arguments` vector
    ...
end
```

Accepts a vector of (name, value) elements, specname, name prefix and proxy type. 
Returns parsed names without prefix and proxied values

See also: [`@rule`](@ref)
"""
function call_rule_macro_parse_fn_args(inputs; specname, prefix, proxy)
    finputs = filter((i) -> startswith(string(first(i)), string(prefix)), inputs)

    lprefix = length(string(prefix))
    names   = map(first, finputs)
    values  = map(last, finputs)

    @assert all((n) -> length(string(n)) > lprefix, names) || error("Empty $(specname) name found in arguments")

    # ManyOf are special cases
    function apply_proxy(any, proxy)
        if any isa Expr && any.head === :call && (any.args[1] === :ManyOf || any.args[1] == :(ReactiveMP.ManyOf))
            argsvar = gensym(:ManyOf)
            return quote
                let
                    local $argsvar = ($(any.args[2:end]...),)
                    if length($argsvar) === 1 && first($argsvar) isa Tuple
                        ReactiveMP.ManyOf(map(element -> $(apply_proxy(:element, proxy)), first($argsvar)))
                    else
                        ReactiveMP.ManyOf(($(map(v -> apply_proxy(v, proxy), any.args[2:end])...),))
                    end
                end
            end
            return :(ReactiveMP.ManyOf(($(map(v -> apply_proxy(v, proxy), any.args[2:end])...),)))
        end
        return :($(proxy)($any, false, false, nothing))
    end

    names_arg  = isempty(names) ? :nothing : :(Val{$(Expr(:tuple, map(n -> QuoteNode(Symbol(string(n)[(lprefix + 1):end])), names)...))}())
    values_arg = isempty(names) ? :nothing : :($(map(v -> apply_proxy(v, proxy), values)...),)

    return names_arg, values_arg
end

# This trait indicates that a node reference is required for a proper rule execution 
# Most of the message passing update rules do not require a node reference
# An example of a rule that requires a node is the `delta`, that needs the node function
struct CallRuleNodeRequired end

# This trait indicates that a node reference is not required for a proper rule execution 
# This is used by default
struct CallRuleNodeNotRequired end

"""
    call_rule_is_node_required(fformtype)

Returns either `CallRuleNodeRequired()` or `CallRuleNodeNotRequired()` depending on if a specific 
`fformtype` requires an access to the corresponding node in order to compute a message update rule.
Returns `CallRuleNodeNotRequired()` for all known functional forms by default and `CallRuleNodeRequired()` for all unknown functional forms.
"""
call_rule_is_node_required(fformtype) = call_rule_is_node_required(is_predefined_node(fformtype), fformtype)

call_rule_is_node_required(::PredefinedNodeFunctionalForm, fformtype) = CallRuleNodeNotRequired()
call_rule_is_node_required(::UndefinedNodeFunctionalForm, fformtype) = CallRuleNodeRequired()

# Returns the `node` if it is required for a rule, otherwise returns `nothing`
node_if_required(fformtype, node) = node_if_required(call_rule_is_node_required(fformtype), node)

node_if_required(::CallRuleNodeRequired, node) = node
node_if_required(::CallRuleNodeNotRequired, node) = nothing

"""
    call_rule_create_node(::Type{ NodeType }, fformtype)

Creates a node object that will be used inside `@call_rule` macro. 
"""
function call_rule_make_node(fformtype, nodetype, meta)
    return call_rule_make_node(call_rule_is_node_required(nodetype), fformtype, nodetype, meta)
end

function call_rule_make_node(::CallRuleNodeRequired, fformtype, nodetype, meta)
    return error("Missing implementation for the `call_rule_make_node`. Cannot create a node of type `$nodetype` for the call rule routine.")
end

function call_rule_make_node(::CallRuleNodeNotRequired, fformtype, nodetype, meta)
    return nothing
end

function call_rule_macro_construct_on_arg(on_type, on_index::Nothing)
    bottomtype = MacroHelpers.bottom_type(on_type)
    if @capture(bottomtype, Val{R_})
        return :(Val($R))
    else
        error("Internal indexed call rule error: Invalid `on_type` in the `call_rule_macro_construct_on_arg` function.")
    end
end

function call_rule_macro_construct_on_arg(on_type, on_index::Int)
    bottomtype = MacroHelpers.bottom_type(on_type)
    if @capture(bottomtype, Tuple{Val{R_}, Int})
        return :((Val($R), $on_index))
    else
        error("Internal indexed call rule error: Invalid `on_type` in the `call_rule_macro_construct_on_arg` function.")
    end
end

function rule_function_expression(body::Function, fuppertype, on_type, vconstraint, m_names, m_types, q_names, q_types, metatype, whereargs)
    addonsvar = gensym(:addons)
    nodevar = gensym(:node)
    return quote
        function ReactiveMP.rule(
            fform::$(fuppertype),
            on::$(on_type),
            vconstraint::$(vconstraint),
            messages_names::$(m_names),
            messages::$(m_types),
            marginals_names::$(q_names),
            marginals::$(q_types),
            meta::$(metatype),
            $(addonsvar),
            $(nodevar)
        ) where {$(whereargs...)}
            local getnode = () -> $nodevar
            local getnodefn = (args...) -> ReactiveMP.nodefunction($nodevar, args...)
            local getaddons = () -> $addonsvar
            $(body())
        end
    end
end

function marginalrule_function_expression(body::Function, fuppertype, on_type, m_names, m_types, q_names, q_types, metatype, whereargs)
    nodevar = gensym(:node)
    return quote
        function ReactiveMP.marginalrule(
            fform::$(fuppertype), on::$(on_type), messages_names::$(m_names), messages::$(m_types), marginals_names::$(q_names), marginals::$(q_types), meta::$(metatype), $nodevar
        ) where {$(whereargs...)}
            local getnode = () -> $nodevar
            local getnodefn = (args...) -> ReactiveMP.nodefunction($nodevar, args...)
            $(body())
        end
    end
end

import .MacroHelpers

"""
    @rule NodeType(:Edge, Constraint) (Arguments..., [ meta::MetaType ]) = begin
        # rule body
        return ...
    end

The `@rule` macro help to define new methods for the `rule` function. It works particularly well in combination with the `@node` macro.
It has a specific structure, which must specify:

- `NodeType`: must be a valid Julia type. If some attempt to define a rule for a Julia function (for example `+`), use `typeof(+)`
- `Edge`: edge label, usually edge labels are defined with the `@node` macro
- `Constrain`: DEPRECATED, please just use the `Marginalisation` label
- `Arguments`: defines a list of the input arguments for the rule
    - `m_*` prefix indicates that the argument is of type `Message` from the edge `*`
    - `q_*` prefix indicates that the argument is of type `Marginal` on the edge `*`
- `Meta::MetaType` - optionally, a user can specify a `Meta` object of type `MetaType`. 
  This can be useful is some attempts to try different rules with different approximation methods or if the rule itself requires some temporary storage or cache. 
  The default meta is `nothing`.


Here are various examples of the `@rule` macro usage:

1. Belief-Propagation (or Sum-Product) message update rule for the `NormalMeanVariance` node  toward the `:μ` edge with the `Marginalisation` constraint.
   Input arguments are `m_out` and `m_v`, which are the messages from the corresponding edges `out` and `v` and have the type `PointMass`.

```julia
@rule NormalMeanVariance(:μ, Marginalisation) (m_out::PointMass, m_v::PointMass) = NormalMeanVariance(mean(m_out), mean(m_v))
```

2. Mean-field message update rule for the `NormalMeanVariance` node towards the `:μ` edge with the `Marginalisation` constraint.
   Input arguments are `q_out` and `q_v`, which are the marginals on the corresponding edges `out` and `v` of type `Any`.

```julia
@rule NormalMeanVariance(:μ, Marginalisation) (q_out::Any, q_v::Any) = NormalMeanVariance(mean(q_out), mean(q_v))
```


3. Structured Variational message update rule for the `NormalMeanVariance` node towards the `:out` edge with the `Marginalisation` constraint.
   Input arguments are `m_μ`, which is a message from the `μ` edge of type `UnivariateNormalDistributionsFamily`, and `q_v`, which is a marginal on the `v` edge of type `Any`.

```julia
@rule NormalMeanVariance(:out, Marginalisation) (m_μ::UnivariateNormalDistributionsFamily, q_v::Any) = begin
    m_μ_mean, m_μ_cov = mean_cov(m_μ)
    return NormalMeanVariance(m_μ_mean, m_μ_cov + mean(q_v))
end
```


See also: [`rule`](@ref), [`marginalrule`](@ref), [`@marginalrule`], [`@call_rule`](@ref)
"""
macro rule(fform, lambda)
    @capture(fform, fformtype_(on_, vconstraint_, options__)) ||
        error("Error in macro. Functional form specification should in the form of 'fformtype_(on_, vconstraint_, options__)'")

    @capture(lambda, (args_ where {whereargs__} = body_) | (args_ = body_)) || error("Error in macro. Lambda body specification is incorrect")

    @capture(args, (inputs__, meta::metatype_) | (inputs__,)) || error("Error in macro. Lambda body arguments specification is incorrect")

    fuppertype                       = MacroHelpers.upper_type(fformtype)
    on_type, on_index, on_index_init = rule_macro_parse_on_tag(on)
    whereargs                        = whereargs === nothing ? [] : whereargs
    metatype                         = metatype === nothing ? :Nothing : metatype

    options = map(options) do option
        @capture(option, name_ = value_) || error("Error in macro. Option specification '$(option)' is incorrect")
        return (name, value)
    end

    inputs = map(inputs) do input
        @capture(input, iname_::itype_) || error("Error in macro. Input $(input) is incorrect")
        return (iname, itype)
    end

    rule_macro_check_fn_args(inputs; allowed_inputs = (:meta,), allowed_prefixes = (:m_, :q_))

    m_names, m_types, m_init_block = rule_macro_parse_fn_args(inputs; specname = :messages, prefix = :m_, proxy = :(ReactiveMP.Message))
    q_names, q_types, q_init_block = rule_macro_parse_fn_args(inputs; specname = :marginals, prefix = :q_, proxy = :(ReactiveMP.Marginal))

    output = quote
        $(
            rule_function_expression(fuppertype, on_type, vconstraint, m_names, m_types, q_names, q_types, metatype, whereargs) do
                return quote
                    local _addons = getaddons()
                    # This trick allows us to use arbitrary control-flow logic
                    # inside rules, e.g. if-else-returns etc, however 
                    # it makes it not-type-stable with respect to addons
                    # on my (bvdmitri) benchmarks it accounted for 2-3% slowdown
                    # when using addons, which is IMO acceptable, but can be changed 
                    # in the future by banning return statements from the `@rule` macro
                    # I'm against of manually removing return statements as 
                    # it is very hard to implement correctly, I would rather make it more stable 
                    # when fast but error-prone
                    # Another way to speed-up this part a little bit would be to refactor addons 
                    # in such a way that their structure is always known to the compiler and type stable
                    local _messagebody = () -> begin
                        $(on_index_init)
                        $(m_init_block...)
                        $(q_init_block...)
                        $(body)
                    end
                    local _message = _messagebody()
                    return _message, _addons
                end
            end
        )
    end

    return esc(output)
end

"""
    @call_rule NodeType(:edge, Constraint) (argument1 = value1, argument2 = value2, ..., [ meta = ..., addons = ... ])

The `@call_rule` macro helps to call the `rule` method with an easier syntax. 
The structure of the macro is almost the same as in the `@rule` macro, but there is no `begin ... end` block, but instead each argument must have a specified value with the `=` operator.

The `@call_rule` accepts optional list of options before the functional form specification, for example:

```julia
@call_rule [ return_addons = true ] NodeType(:edge, Constraint) (argument1 = value1, argument2 = value2, ..., [ meta = ..., addons = ... ])
```

The list of available options is:

- `return_addons` - forces the `@call_rule` to return the tuple of `(result, addons)`
- `fallback` - specifies the fallback rule to use in case the rule is not defined for the given `NodeType` and specified arguments

See also: [`@rule`](@ref), [`rule`](@ref), [`@call_marginalrule`](@ref)
"""
macro call_rule(options, fform, args)
    return call_rule_expression(options, fform, args)
end

macro call_rule(fform, args)
    return call_rule_expression(nothing, fform, args)
end

function call_rule_expression(options, fform, args)
    @capture(fform, fformtype_(on_, vconstraint_)) || error("Error in macro. Functional form specification should in the form of 'fformtype_(on_, vconstraint_)'")

    @capture(args, (inputs__, meta = meta_, addons = addons_) | (inputs__, addons = addons_) | (inputs__, meta = meta_) | (inputs__,)) ||
        error("Error in macro. Arguments specification is incorrect")

    fuppertype                       = MacroHelpers.upper_type(fformtype)
    fbottomtype                      = MacroHelpers.bottom_type(fformtype)
    on_type, on_index, on_index_init = rule_macro_parse_on_tag(on)
    node                             = :(ReactiveMP.call_rule_make_node($fformtype, $fbottomtype, $meta))

    inputs = map(inputs) do input
        @capture(input, iname_ = ivalue_) || error("Error in macro. Argument $(input) is incorrect")
        return (iname, ivalue)
    end

    rule_macro_check_fn_args(inputs; allowed_inputs = (:meta,), allowed_prefixes = (:m_, :q_))

    m_names_arg, m_values_arg = call_rule_macro_parse_fn_args(inputs; specname = :messages, prefix = :m_, proxy = :(ReactiveMP.Message))
    q_names_arg, q_values_arg = call_rule_macro_parse_fn_args(inputs; specname = :marginals, prefix = :q_, proxy = :(ReactiveMP.Marginal))

    on_arg = call_rule_macro_construct_on_arg(on_type, on_index)

    # Options
    # Option 1. Modifies the output of the `@call_rule` macro and returns a tuple of the result and the enabled addons
    return_addons = false
    fallback = nothing

    if !isnothing(options)
        @capture(options, [voptions__]) || error("Error in macro. Options should be in a form of `[ option1 = value1, ... ]`, got $(options).")
        foreach(voptions) do option
            @capture(option, key_ = value_) || error("Error in macro. An options should be in a form of `option = value`, got $(option).")
            if key === :return_addons
                return_addons = Bool(value)
            elseif key === :fallback
                fallback = value
            else
                @warn "Unknown option in the `@call_rule` macro: $(option)"
            end
        end
    end

    __rule_result_sym = gensym(:call_rule_result)
    __distribution_sym = gensym(:call_rule_distribution)
    __addons_sym = gensym(:call_rule_addons)

    call = quote
        local $__rule_result_sym = ReactiveMP.rule($fbottomtype, $on_arg, $(vconstraint)(), $m_names_arg, $m_values_arg, $q_names_arg, $q_values_arg, $meta, $addons, $node)
        if ($__rule_result_sym) isa ReactiveMP.RuleMethodError && !isnothing($fallback)
            $__rule_result_sym = $(fallback)($fbottomtype, $on_arg, $(vconstraint)(), $m_names_arg, $m_values_arg, $q_names_arg, $q_values_arg, $meta, $addons, $node)
        elseif ($__rule_result_sym) isa ReactiveMP.RuleMethodError
            throw($__rule_result_sym)
        end
        local $(__distribution_sym), $(__addons_sym) = $(__rule_result_sym)
    end

    output = if !return_addons
        :($call; $__distribution_sym)
    else
        :($call)
    end

    return esc(output)
end

"""
    @marginalrule NodeType(:Cluster) (Arguments..., [ meta::MetaType ]) = begin
        # rule body
        return ...
    end

The `@marginalrule` macro help to define new methods for the `marginalrule` function. It works particularly well in combination with the `@node` macro.
It has a specific structure, which must specify:

- `NodeType`: must be a valid Julia type. If some attempt to define a rule for a Julia function (for example `+`), use `typeof(+)`
- `Cluster`: edge cluster that contains joined edge labels with the `_` symbol. Usually edge labels are defined with the `@node` macro
- `Arguments`: defines a list of the input arguments for the rule
    - `m_*` prefix indicates that the argument is of type `Message` from the edge `*`
    - `q_*` prefix indicates that the argument is of type `Marginal` on the edge `*`
- `Meta::MetaType` - optionally, a user can specify a `Meta` object of type `MetaType`. 
  This can be useful is some attempts to try different rules with different approximation methods or if the rule itself requires some temporary storage or cache. 
  The default meta is `nothing`.

The `@marginalrule` can return a `NamedTuple` in the `return` statement. This would indicate some variables in the joint marginal 
for the `Cluster` are independent and the joint itself is factorised. For example if some attempts to compute a marginal for the `q(x, y)` it is possible to return
`(x = ..., y = ...)` as the result of the computation to indicate that `q(x, y) = q(x)q(y)`.

Here are various examples of the `@marginalrule` macro usage:

1. Marginal computation rule around the `NormalMeanPrecision` node for the `q(out, μ)`. The rule accepts arguments `m_out` and `m_μ`, which are the messages 
from the `out` and `μ` edges respectively, and `q_τ` which is the marginal on the edge `τ`.

```julia
@marginalrule NormalMeanPrecision(:out_μ) (m_out::UnivariateNormalDistributionsFamily, m_μ::UnivariateNormalDistributionsFamily, q_τ::Any) = begin
    xi_out, W_out = weightedmean_precision(m_out)
    xi_μ, W_μ     = weightedmean_precision(m_μ)

    W_bar = mean(q_τ)

    W  = [W_out+W_bar -W_bar; -W_bar W_μ+W_bar]
    xi = [xi_out; xi_μ]

    return MvNormalWeightedMeanPrecision(xi, W)
end
```

2. Marginal computation rule around the `NormalMeanPrecision` node for the `q(out, μ)`. The rule accepts arguments `m_out` and `m_μ`, which are the messages from the 
`out` and `μ` edges respectively, and `q_τ` which is the marginal on the edge `τ`. In this example the result of the computation is a `NamedTuple`

```julia
@marginalrule NormalMeanPrecision(:out_μ) (m_out::PointMass, m_μ::UnivariateNormalDistributionsFamily, q_τ::Any) = begin
    return (out = m_out, μ = prod(ClosedProd(), NormalMeanPrecision(mean(m_out), mean(q_τ)), m_μ))
end
```

"""
macro marginalrule(fform, lambda)
    @capture(fform, fformtype_(on_)) || error("Error in macro. Functional form specification should in the form of 'fformtype_(on_)'")

    @capture(lambda, (args_ where {whereargs__} = body_) | (args_ = body_)) || error("Error in macro. Lambda body specification is incorrect")

    @capture(args, (inputs__, meta::metatype_) | (inputs__,)) || error("Error in macro. Lambda body arguments specification is incorrect")

    fuppertype                       = MacroHelpers.upper_type(fformtype)
    on_type, on_index, on_index_init = rule_macro_parse_on_tag(on)
    whereargs                        = whereargs === nothing ? [] : whereargs
    metatype                         = metatype === nothing ? :Any : metatype

    inputs = map(inputs) do input
        @capture(input, iname_::itype_) || error("Error in macro. Input $(input) is incorrect")
        return (iname, itype)
    end

    rule_macro_check_fn_args(inputs; allowed_inputs = (:meta,), allowed_prefixes = (:m_, :q_))

    m_names, m_types, m_init_block = rule_macro_parse_fn_args(inputs; specname = :messages, prefix = :m_, proxy = :(ReactiveMP.Message))
    q_names, q_types, q_init_block = rule_macro_parse_fn_args(inputs; specname = :marginals, prefix = :q_, proxy = :(ReactiveMP.Marginal))

    output = quote
        $(
            marginalrule_function_expression(fuppertype, on_type, m_names, m_types, q_names, q_types, metatype, whereargs) do
                return quote
                    $(on_index_init)
                    $(m_init_block...)
                    $(q_init_block...)
                    $(body)
                end
            end
        )
    end

    return esc(output)
end

"""
    @call_marginalrule NodeType(:edge) (argument1 = value1, argument2 = value2, ..., [ meta = ... ])

The `@call_marginalrule` macro helps to call the `marginalrule` method with an easier syntax. 
The structure of the macro is almost the same as in the `@marginalrule` macro, but there is no `begin ... end` block, 
but instead each argument must have a specified value with the `=` operator.

See also: [`@marginalrule`](@ref), [`marginalrule`](@ref), [`@call_rule`](@ref)
"""
macro call_marginalrule(fform, args)
    @capture(fform, fformtype_(on_)) || error("Error in macro. Functional form specification should in the form of 'fformtype_(on_)'")

    @capture(args, (inputs__, meta = meta_) | (inputs__,)) || error("Error in macro. Arguments specification is incorrect")

    fuppertype                       = MacroHelpers.upper_type(fformtype)
    fbottomtype                      = MacroHelpers.bottom_type(fformtype)
    on_type, on_index, on_index_init = rule_macro_parse_on_tag(on)
    node                             = :(ReactiveMP.call_rule_make_node($fformtype, $fbottomtype, $meta))

    inputs = map(inputs) do input
        @capture(input, iname_ = ivalue_) || error("Error in macro. Argument $(input) is incorrect")
        return (iname, ivalue)
    end

    rule_macro_check_fn_args(inputs; allowed_inputs = (:meta,), allowed_prefixes = (:m_, :q_))

    m_names_arg, m_values_arg = call_rule_macro_parse_fn_args(inputs; specname = :messages, prefix = :m_, proxy = :(ReactiveMP.Message))
    q_names_arg, q_values_arg = call_rule_macro_parse_fn_args(inputs; specname = :marginals, prefix = :q_, proxy = :(ReactiveMP.Marginal))

    on_arg = call_rule_macro_construct_on_arg(on_type, on_index)

    output = quote
        ReactiveMP.marginalrule($fbottomtype, $on_arg, $m_names_arg, $m_values_arg, $q_names_arg, $q_values_arg, $meta, $node)
    end

    return esc(output)
end

## Testing utilities

# (bvdmitri) These are cryptic manually constructed expressions are needed to call a `macro` within another `macro`, 
# but these must be prefixed with the `ReactiveMP` module. There might be a more elegant way to do the same, 
# but I couldn't find one
const CallRuleMacroFnExpr = Expr(:., :ReactiveMP, QuoteNode(Symbol("@call_rule")))
const CallMarginalRuleMacroFnExpr = Expr(:., :ReactiveMP, QuoteNode(Symbol("@call_marginalrule")))
const TestRulesDefaultTestCallback = quote
    (expression) -> @test expression
end

"""
    @test_rules [options] rule [ test_entries... ]

The `@test_rules` macro generates test cases for message update rules for probabilistic programming models that follow the "message passing" paradigm. It takes a rule specification as input and generates a set of tests based on that specification. This macro is provided by `ReactiveMP`.

Note: The `Test` module must be imported explicitly. The `@test_rules` macro tries to use the `@test` macro, which must be defined globally.

## Arguments

The macro takes three arguments:

- `options`: An optional argument that specifies the options for the test generation process. See below for details.
- `rule`: A rule specification in the same format as the `@rule` macro, e.g. `Beta(:out, Marginalisation)` or `NormalMeanVariance(:μ, Marginalisation)`.
- `test_entries`: An array of named tuples `(input = ..., output = ...)`. The `input` entry has the same format as the input for the `@rule` macro. The `output` entry specifies the expected output.

## Options

The following options are available:

- `check_type_promotion`: By default, this option is set to `false`. If set to `true`, the macro generates an extensive list of extra tests that aim to check the correct type promotion within the tests. For example, if all inputs are of type `Float32`, then the expected output should also be of type `Float32`. See the `paramfloattype` and `convert_paramfloattype` functions for details.
- `atol`: Sets the desired accuracy for the tests. The tests use the `custom_rule_isapprox` function from `ReactiveMP` to check if outputs are approximately the same. This argument can be either a single number or an array of `key => value` pairs.
- `extra_float_types`: A set of extra float types to be used in the `check_type_promotion` tests. This argument has no effect if `check_type_promotion` is set to `false`.

The default values for the `atol` option are:

- `Float32`: `1e-4`
- `Float64`: `1e-6`
- `BigFloat`: `1e-8`

## Examples

```julia

@test_rules [check_type_promotion = true] Beta(:out, Marginalisation) [
    (input = (m_a = PointMass(1.0), m_b = PointMass(2.0)), output = Beta(1.0, 2.0)),
    (input = (m_a = PointMass(2.0), m_b = PointMass(2.0)), output = Beta(2.0, 2.0)),
    (input = (m_a = PointMass(3.0), m_b = PointMass(3.0)), output = Beta(3.0, 3.0))
]

@test_rules [check_type_promotion = true] Beta(:out, Marginalisation) [
    (input = (q_a = PointMass(1.0), q_b = PointMass(2.0)), output = Beta(1.0, 2.0)),
    (input = (q_a = PointMass(2.0), q_b = PointMass(2.0)), output = Beta(2.0, 2.0)),
    (input = (q_a = PointMass(3.0), q_b = PointMass(3.0)), output = Beta(3.0, 3.0))
]
```

See also: [`ReactiveMP.@test_marginalrules`](@ref)
"""
macro test_rules end

"""
    @test_marginalrules [options] rule [ test_entries... ]

Effectively the same as `@test_rules`, but for marginal computation rules. See the documentation for `@test_rules` for more info.

See also: [`ReactiveMP.@test_rules`](@ref)
"""
macro test_marginalrules end

macro test_rules(rule_specification, tests)
    return ReactiveMP.test_rules_generate(TestRulesDefaultTestCallback, CallRuleMacroFnExpr, :([]), rule_specification, tests)
end

macro test_rules(options, rule_specification, tests)
    return ReactiveMP.test_rules_generate(TestRulesDefaultTestCallback, CallRuleMacroFnExpr, options, rule_specification, tests)
end

macro test_rules(call_test_fn, options, rule_specification, tests)
    return ReactiveMP.test_rules_generate(call_test_fn, CallRuleMacroFnExpr, options, rule_specification, tests)
end

macro test_marginalrules(rule_specification, tests)
    return ReactiveMP.test_rules_generate(TestRulesDefaultTestCallback, CallMarginalRuleMacroFnExpr, :([]), rule_specification, tests)
end

macro test_marginalrules(options, rule_specification, tests)
    return ReactiveMP.test_rules_generate(TestRulesDefaultTestCallback, CallMarginalRuleMacroFnExpr, options, rule_specification, tests)
end

macro test_marginalrules(call_test_fn, options, rule_specification, tests)
    return ReactiveMP.test_rules_generate(call_test_fn, CallMarginalRuleMacroFnExpr, options, rule_specification, tests)
end

function test_rules_generate(call_test_fn, call_macro_fn, options, rule_specification, tests)
    testsblock = :__tests_block
    configuration = :__configuration
    configuration_opts = test_rules_parse_configuration(configuration, options)
    test_entries = convert(Vector{TestRuleEntry}, tests)

    # We guard the `@test` macro within the callback function, such that it 
    # does not generate a lot of garbage in the macro (makes it more efficient)
    test_fn = :__test_fn

    default_tests = Expr(:block)
    default_tests.args = map(test_entries) do test_entry
        return test_rules_generate_testset(test_entry, test_fn, call_macro_fn, rule_specification, configuration)
    end

    # Extra tests for type promotion, could be turned off if `check_type_promotion = false`
    type_promotion_T = :__promoted_T
    type_promotion_tests = map(test_entry -> test_rules_convert_paramfloattype_for_test_entry(test_entry, type_promotion_T), test_entries)
    type_promotion_block = Expr(:block)
    type_promotion_block.args = map(Iterators.flatten(type_promotion_tests)) do promoted_test_entry
        return test_rules_generate_testset(promoted_test_entry, test_fn, call_macro_fn, rule_specification, configuration)
    end

    output = quote
        let $configuration = ReactiveMP.TestRulesConfiguration()
            # Insert configuration options
            $configuration_opts
            # Insert test callback 
            local $test_fn = $call_test_fn
            # Insert generated tests, these tests are returned by default
            local $(testsblock) = $default_tests

            # Check if the `check_type_promotion` is true and
            # perform extra test set if required
            if ReactiveMP.check_type_promotion($configuration)
                for $(type_promotion_T) in ReactiveMP.extra_float_types($(configuration))
                    $type_promotion_block
                end
            end

            $(testsblock)
        end
    end

    return esc(output)
end

Base.@kwdef mutable struct TestRulesConfiguration
    check_type_promotion::Bool = false
    float_tolerance::Dict = Dict(Float32 => 1e-4, Float64 => 1e-6, BigFloat => 1e-8)
    extra_float_types::Vector = [Float32, Float64, BigFloat]
end

const DefaultFloatTolerance = 1e-6

check_type_promotion(configuration::TestRulesConfiguration)::Bool = configuration.check_type_promotion
check_type_promotion!(configuration::TestRulesConfiguration, check::Bool) = configuration.check_type_promotion = check

float_tolerance(configuration::TestRulesConfiguration) = configuration.float_tolerance
float_tolerance(configuration::TestRulesConfiguration, ::Type{T}) where {T} = get(() -> DefaultFloatTolerance, float_tolerance(configuration), T)

float_tolerance!(configuration::TestRulesConfiguration, ::Type{T}, atol::Number) where {T} = configuration.float_tolerance[T] = atol
float_tolerance!(configuration::TestRulesConfiguration, atol::Number) = foreach(((key, _),) -> float_tolerance!(configuration, key, atol), float_tolerance(configuration))
float_tolerance!(configuration::TestRulesConfiguration, atol::AbstractArray) = foreach(((key, value),) -> float_tolerance!(configuration, key, value), atol)

extra_float_types(configuration::TestRulesConfiguration) = configuration.extra_float_types
extra_float_types!(configuration::TestRulesConfiguration, types) = configuration.extra_float_types = types

# used in the `@test_rules/@test_marginalrules`
function test_rules_parse_configuration(configuration::Symbol, options::Expr)
    @capture(options, [option_entries__]) || error("Cannot parse the test configuration. The options must be an array of `key = value` pairs.")

    block = Expr(:block)
    block.args = map(option_entries) do entry
        @capture(entry, key_ = value_) || error("Cannot parse the test configuration. The options must be an array of `key = value` pairs.")

        if key === :check_type_promotion
            return :(ReactiveMP.check_type_promotion!($configuration, convert(Bool, $value)))
        elseif key === :atol
            return :(ReactiveMP.float_tolerance!($configuration, $value))
        elseif key === :extra_float_types
            return :(ReactiveMP.extra_float_types!($configuration, $value))
        else
            error("Unknown option for the test configuration $(key)")
        end
    end

    return block
end

# Represents a specification for test rules `input = (...)` keyword argument
# Store arguments as vector of key-value pairs and the meta specification
struct TestRuleEntryInputSpecification
    arguments::Vector{Pair{Symbol, Any}}
    meta::Any
end

Base.:(==)(left::TestRuleEntryInputSpecification, right::TestRuleEntryInputSpecification) = (left.arguments == right.arguments) && (left.meta == right.meta)

Base.copy(entry::TestRuleEntryInputSpecification) = TestRuleEntryInputSpecification(copy(entry.arguments), entry.meta) # no need to copy `meta`
Base.values(entry::TestRuleEntryInputSpecification) = Base.Generator((arg) -> arg.second, entry.arguments)

# Convert the `TestRuleEntryInputSpecification` back into the `Expr` form, e.g `(m_x = ..., q_y = ..., meta = ...)`
function rule_macro_convert_to_expr(test_entry::TestRuleEntryInputSpecification)
    tuple = Expr(:tuple)
    tuple.args = map((arg) -> Expr(:(=), arg.first, arg.second), test_entry.arguments)
    if !isnothing(test_entry.meta)
        push!(tuple.args, Expr(:(=), :meta, test_entry.meta))
    end
    return tuple
end

# This function parses expressions of the form
# (key1 = value1, key2 = value2, ..., [ meta = ... ]) 
# and returns `TestRuleEntryInputSpecification`
function Base.convert(::Type{TestRuleEntryInputSpecification}, input::Expr)
    @capture(input, (pairs__,)) || error("Cannot parse the `input` specification: $(input). Should be in a form of the `NamedTuple`.")

    arguments = Pair{Symbol, Any}[]
    meta = nothing

    for pair in pairs
        @capture(pair, key_ = value_) || error("Cannot parse an argument of the `input` specification: $(pair). Should be in a form of the `key = value` expression.")
        if key === :meta # Reserved for the `meta` specification
            meta = value
        else
            push!(arguments, key => value)
        end
    end

    return TestRuleEntryInputSpecification(arguments, meta)
end

struct TestRuleEntry
    input::TestRuleEntryInputSpecification
    output::Any
end

# Convert the `TestRuleEntry` back into the `Expr` form, e.g `(input = ..., output = ...)`
function rule_macro_convert_to_expr(test_entry::TestRuleEntry)
    return Expr(:tuple, Expr(:(=), :input, rule_macro_convert_to_expr(test_entry.input)), Expr(:(=), :output, test_entry.output))
end

# This function takes a `test` parameter which is expected to be an expression of single test entry.
# The test entry should be an expression of named tuple with an `input` and an `output` field.
# The function returns an instance of the `TestRuleEntry` structure
function Base.convert(::Type{TestRuleEntry}, test::Expr)
    @capture(test, (input = input_, output = output_)) ||
        error("Invalid test entry specification. Test entry should be in the form of a named tuple `(input = ..., output = ...)`.")
    p_input = convert(TestRuleEntryInputSpecification, input)
    p_output = output
    return TestRuleEntry(p_input, p_output)
end

function Base.convert(::Type{Vector{TestRuleEntry}}, tests::Expr)
    @capture(tests, [test_entries__]) || error("Invalid tests specification. Test sequence should be in the form of an array.")
    return map(test_entries) do test_entry
        return convert(TestRuleEntry, test_entry)
    end
end

# This function creates a set of type promotion tests for rules
# First it creates a set of all possible subsets of input `key = value` pairs
# Then for each subset it convert the `key = value` pair to a specific float type (e.g. Float32)
# The resulting float type of the rule is expected to be the same as the promoted type of 
# the all `key = value` pairs after conversion
function test_rules_convert_paramfloattype_for_test_entry(test_entry::TestRuleEntry, eltype)
    input = test_entry.input
    output = test_entry.output

    # See the mathematical definition of powerset
    combinations = powerset(1:length(input.arguments))

    return Base.Generator(combinations) do combination
        cinput = copy(input) # `meta` is not copied
        for index in combination
            cinput.arguments[index] = (cinput.arguments[index].first => test_rules_convert_paramfloattype(cinput.arguments[index].second, eltype))
        end
        cvalues = values(cinput)
        coutput_eltype = test_rules_promote_paramfloattype(cvalues)
        coutput = test_rules_convert_paramfloattype(output, coutput_eltype)
        return TestRuleEntry(cinput, coutput)
    end
end

function test_rules_generate_testset(test_entry::TestRuleEntry, invoke_test_fn, call_macro_fn, rule_specification, configuration)
    # `nothing` here is a `LineNumberNode`, macrocall expects a `line` number, but we do not have it here
    actual_inputs = rule_macro_convert_to_expr(test_entry.input)
    actual_output = Expr(:macrocall, call_macro_fn, nothing, rule_specification, actual_inputs)
    expected_output = test_entry.output
    rule_spec_str = "$rule_specification"
    rule_inputs_str = "$actual_inputs"
    generated = quote
        let invoke_test_fn = $invoke_test_fn, expected_output = $expected_output, actual_output = $actual_output, rule_spec_str = $rule_spec_str, rule_inputs_str = $rule_inputs_str
            local _T = ReactiveMP.promote_paramfloattype(actual_output, expected_output)
            local _tolerance = ReactiveMP.float_tolerance($configuration, _T)
            local _isapprox = ReactiveMP.custom_rule_isapprox(actual_output, expected_output; atol = _tolerance)
            local _isequal_typeof = ReactiveMP.BayesBase.isequal_typeof(actual_output, expected_output)

            if !_isapprox || !_isequal_typeof
                ReactiveMP.test_rules_failed_warning(rule_spec_str, rule_inputs_str, expected_output, actual_output)
            end

            # We should not put `@test` within the aut-generated macro, because it allocates a lot of garbage
            invoke_test_fn(_isapprox && _isequal_typeof)
        end
    end
    return generated
end

# We should not put `@warn` withtin the auto-generated macro, because it allocates
# a lot of garbage code
function test_rules_failed_warning(rule_specification, rule_inputs, expected_output, actual_output)
    @warn """
        Testset for rule $(rule_specification) has failed!
        Inputs: $(rule_inputs)
        Expected output: $(expected_output)
        Actual output: $(actual_output)
        Expected type: $(typeof(expected_output))
        Actual type: $(typeof(actual_output))
    """
end

# This function converts `key = value` pair to `key = convert_paramfloattype(eltype, value)`
# Calls recursively for tuples and for `ManyOf` structures, which are tuple-like but for rules
function test_rules_convert_paramfloattype(expression, eltype)
    if @capture(expression, (entries__,))
        return :(($(map(entry -> ReactiveMP.test_rules_convert_paramfloattype(entry, eltype), entries)...),))
    elseif @capture(expression, (ManyOf(entries__)) | (ReactiveMP.ManyOf(entries__)))
        return :(ManyOf($(map(entry -> ReactiveMP.test_rules_convert_paramfloattype(entry, eltype), entries)...)))
    elseif @capture(expression, key_ = value_)
        return :($key = $(ReactiveMP.test_rules_convert_paramfloattype(value, eltype)))
    else
        return :(ReactiveMP.BayesBase.convert_paramfloattype($eltype, $expression))
    end
end

# This functiona takes an an array of `values` and generates an expression
# that returns a promoted paramfloattype of those `values`
function test_rules_promote_paramfloattype(values)
    cvalues = map(values) do value
        if @capture(value, (ManyOf(entries__) | (ReactiveMP.ManyOf(entries__))))
            return :(($(entries...),))
        else
            return value
        end
    end
    return :(ReactiveMP.BayesBase.promote_paramfloattype($(cvalues...)))
end

# Error utilities

## Custom approx 

import DomainIntegrals, DomainSets

custom_rule_isapprox(left, right; kwargs...) = isapprox(left, right; kwargs...)
custom_rule_isapprox(left::NamedTuple, right::NamedTuple; kwargs...) = false

function custom_rule_isapprox(left::NamedTuple{K}, right::NamedTuple{K}; kwargs...) where {K}
    _isapprox = true
    for key in keys(left)
        _isapprox = _isapprox && custom_rule_isapprox(left[key], right[key]; kwargs...)
    end
    return _isapprox
end

import BayesBase: AbstractContinuousGenericLogPdf

# These methods are inaccurate and relies on various approximation methods, which may fail in different scenarios
# This should not be used though anywhere in the real code, but only in tests
# Current implementation of `isapprox` method supports only FullSpace and HalfLine domains with limited accuracy
function custom_rule_isapprox(left::AbstractContinuousGenericLogPdf, right::AbstractContinuousGenericLogPdf; kwargs...)
    if (BayesBase.getdomain(left) !== BayesBase.getdomain(right)) ||
        (value_support(typeof(left)) !== value_support(typeof(right))) ||
        (variate_form(typeof(left)) !== variate_form(typeof(right)))
        return false
    end
    return culogpdf__isapprox(BayesBase.getdomain(left), left, right; kwargs...)
end

# https://en.wikipedia.org/wiki/Gauss–Hermite_quadrature
function culogpdf__isapprox(domain::DomainSets.FullSpace, left::AbstractContinuousGenericLogPdf, right::AbstractContinuousGenericLogPdf; kwargs...)
    return isapprox(zero(eltype(domain)), DomainIntegrals.integral(DomainIntegrals.Q_GaussHermite(32), (x) -> exp(x^2) * abs(left(x) - right(x))); kwargs...)
end

# https://en.wikipedia.org/wiki/Gauss–Laguerre_quadrature
function culogpdf__isapprox(domain::DomainSets.HalfLine, left::AbstractContinuousGenericLogPdf, right::AbstractContinuousGenericLogPdf; kwargs...)
    return isapprox(zero(eltype(domain)), DomainIntegrals.integral(DomainIntegrals.Q_GaussLaguerre(32), (x) -> exp(x) * abs(left(x) - right(x))); kwargs...)
end

function culogpdf__isapprox(domain::DomainSets.VcatDomain, left::AbstractContinuousGenericLogPdf, right::AbstractContinuousGenericLogPdf; kwargs...)
    a = clamp.(DomainSets.infimum(domain), -1e5, 1e5)
    b = clamp.(DomainSets.supremum(domain), -1e5, 1e5)
    (I, E) = HCubature.hcubature((x) -> abs(left(x) - right(x)), a, b)
    return isapprox(zero(promote_paramfloattype(left, right)), I; kwargs...) && isapprox(zero(promote_paramfloattype(left, right)), E; kwargs...)
end

function culogpdf__isapprox(domain::DomainSets.FixedIntervalProduct, left::AbstractContinuousGenericLogPdf, right::AbstractContinuousGenericLogPdf; kwargs...)
    a = clamp.(DomainSets.infimum(domain), -1e5, 1e5)
    b = clamp.(DomainSets.supremum(domain), -1e5, 1e5)
    (I, E) = HCubature.hcubature((x) -> abs(left(x) - right(x)), a, b)
    return isapprox(zero(promote_paramfloattype(left, right)), I; kwargs...) && isapprox(zero(promote_paramfloattype(left, right)), E; kwargs...)
end

## Dummy node

mutable struct NodeErrorStub
    counter::Int
end

NodeErrorStub() = NodeErrorStub(0)

interfaceindices(stub::NodeErrorStub, iname::Symbol)                       = (interfaceindex(stub, iname),)
interfaceindices(stub::NodeErrorStub, inames::NTuple{N, Symbol}) where {N} = map(iname -> interfaceindex(stub, iname), inames)

function interfaceindex(stub::NodeErrorStub, iname::Symbol)
    stub.counter = stub.counter + 1
    return stub.counter
end

function getinterfaces(stub::NodeErrorStub)
    return fill(nothing, stub.counter)
end

rule_method_error_extract_fform(f::Function) = string("typeof(", f, ")")
rule_method_error_extract_fform(f)           = string(f)

rule_method_error_extract_on(::Val{T}) where {T}              = string(":", T)
rule_method_error_extract_on(::Tuple{Val{T}, Int}) where {T}  = string("(", rule_method_error_extract_on(Val{T}()), ", k)")
rule_method_error_extract_on(::Tuple{Val{T}, N}) where {T, N} = string("(", rule_method_error_extract_on(Val{T}()), ", ", convert(Int, N), ")")

rule_method_error_extract_vconstraint(something) = typeof(something)

rule_method_error_extract_names(::Val{T}) where {T} = map(sT -> unval(split_underscored_symbol(Val{sT}())), T)
rule_method_error_extract_names(::Nothing)          = ()

rule_method_error_extract_types(t::Tuple)   = map(e -> rule_method_error_type_nameof(typeofdata(e)), t)
rule_method_error_extract_types(t::Nothing) = ()

rule_method_error_type_nameof(something) = nameof(something)

rule_method_error_extract_meta(something) = string("meta::", typeof(something))
rule_method_error_extract_meta(::Nothing) = ""

struct RuleMethodError
    fform
    on
    vconstraint
    mnames
    messages
    qnames
    marginals
    meta
    addons
    node
end

rule(fform, on, vconstraint, mnames, messages, qnames, marginals, meta, addons, __node) = RuleMethodError(
    fform, on, vconstraint, mnames, messages, qnames, marginals, meta, addons, __node
)

function Base.showerror(io::IO, error::RuleMethodError)
    print(io, "RuleMethodError: no method matching rule for the given arguments")

    node = error.node !== nothing ? error.node : NodeErrorStub()

    spec_fform       = rule_method_error_extract_fform(error.fform)
    spec_on          = rule_method_error_extract_on(error.on)
    spec_vconstraint = rule_method_error_extract_vconstraint(error.vconstraint)

    m_names   = rule_method_error_extract_names(error.mnames)
    m_indices = map(n -> TupleTools.maximum(interfaceindices(node, n)), m_names)
    q_names   = rule_method_error_extract_names(error.qnames)
    q_indices = map(n -> TupleTools.maximum(interfaceindices(node, n)), q_names)

    spec_m_names = map(e -> string("m_", join(e, "_")), m_names)
    spec_m_types = rule_method_error_extract_types(error.messages)
    spec_q_names = map(e -> string("q_", join(e, "_")), q_names)
    spec_q_types = rule_method_error_extract_types(error.marginals)

    if isempty(intersect(Set(m_indices), Set(q_indices)))
        spec_m = map(m -> string(m[1], "::", m[2]), zip(spec_m_names, spec_m_types))
        spec_q = map(q -> string(q[1], "::", q[2]), zip(spec_q_names, spec_q_types))

        spec = Vector(undef, 4length(getinterfaces(node)))

        fill!(spec, nothing)

        for (i, j) in enumerate(m_indices)
            spec[2j - 1] = spec_m[i]
        end

        for (i, j) in enumerate(q_indices)
            spec[2j] = spec_q[i]
        end

        filter!(!isnothing, spec)

        arguments_spec = join(spec, ", ")
        meta_spec      = rule_method_error_extract_meta(error.meta)

        possible_fix_definition = """
        @rule $(spec_fform)($spec_on, $spec_vconstraint) ($arguments_spec, $meta_spec) = begin 
            return ...
        end
        """

        println(io, "\n\nPossible fix, define:\n")
        println(io, possible_fix_definition)
        if !isnothing(error.addons)
            println(io, "\n\nEnabled addons: ", error.addons, "\n")
        end

        node_rules = filter(m -> ReactiveMP.get_node_from_rule_method(m) == spec_fform, methods(ReactiveMP.rule))
        println(io, "Alternatively, consider re-specifying model using an existing rule:\n")

        node_message_names = filter(x -> x != ["Nothing"], get_message_names_from_rule_method.(node_rules))
        node_message_types = filter(!isempty, get_message_types_from_rule_method.(node_rules))
        for (m_name, m_type) in zip(node_message_names, node_message_types)
            message_input = [string("m_", n, "::", t) for (n, t) in zip(m_name, m_type)]
            println(io, spec_fform, "(", join(message_input, ", "), ")")
        end

        node_marginal_names = filter(x -> x != ["Nothing"], get_marginal_names_from_rule_method.(node_rules))
        node_marginal_types = filter(!isempty, get_marginal_types_from_rule_method.(node_rules))
        for (m_name, m_type) in zip(node_marginal_names, node_marginal_types)
            marginal_input = [string("q_", n, "::", t) for (n, t) in zip(m_name, m_type)]
            println(io, spec_fform, "(", join(marginal_input, ", "), ")")
        end
        if !isempty(node_marginal_names)
            println(io, "\nNote that for marginal rules (i.e., involving q_*), the order of input types matters.")
        end

    else
        println(io, "\n\n[WARN]: Non-standard rule layout found! Possible fix, define rule with the following arguments:\n")
        println(io, "rule.fform: ", error.fform)
        println(io, "rule.on: ", error.on)
        println(io, "rule.vconstraint: ", error.vconstraint)
        println(io, "rule.mnames: ", error.mnames)
        println(io, "rule.messages: ", error.messages)
        println(io, "rule.qnames: ", error.qnames)
        println(io, "rule.marginals: ", error.marginals)
        println(io, "rule.meta: ", error.meta)
        println(io, "rule.addons: ", error.addons)
    end
end

struct MarginalRuleMethodError
    fform
    on
    mnames
    messages
    qnames
    marginals
    meta
    node
end

marginalrule(fform, on, mnames, messages, qnames, marginals, meta, __node) = throw(MarginalRuleMethodError(fform, on, mnames, messages, qnames, marginals, meta, __node))

function Base.showerror(io::IO, error::MarginalRuleMethodError)
    print(io, "MarginalRuleMethodError: no method matching rule for the given arguments")
    node = error.node !== nothing ? error.node : NodeErrorStub()

    spec_fform = rule_method_error_extract_fform(error.fform)
    spec_on    = rule_method_error_extract_on(error.on)

    m_names   = rule_method_error_extract_names(error.mnames)
    m_indices = map(n -> TupleTools.maximum(interfaceindices(node, n)), m_names)
    q_names   = rule_method_error_extract_names(error.qnames)
    q_indices = map(n -> TupleTools.maximum(interfaceindices(node, n)), q_names)

    spec_m_names = map(e -> string("m_", join(e, "_")), m_names)
    spec_m_types = rule_method_error_extract_types(error.messages)
    spec_q_names = map(e -> string("q_", join(e, "_")), q_names)
    spec_q_types = rule_method_error_extract_types(error.marginals)

    spec_m = map(m -> string(m[1], "::", m[2]), zip(spec_m_names, spec_m_types))
    spec_q = map(q -> string(q[1], "::", q[2]), zip(spec_q_names, spec_q_types))

    spec = Vector(undef, 4length(getinterfaces(node)))

    if isempty(intersect(Set(m_indices), Set(q_indices)))
        fill!(spec, nothing)

        for (i, j) in enumerate(m_indices)
            spec[2j - 1] = spec_m[i]
        end

        for (i, j) in enumerate(q_indices)
            spec[2j] = spec_q[i]
        end

        filter!(!isnothing, spec)

        arguments_spec = join(spec, ", ")
        meta_spec      = rule_method_error_extract_meta(error.meta)

        possible_fix_definition = """
        @marginalrule $(spec_fform)($spec_on) ($arguments_spec, $meta_spec) = begin 
            return ...
        end
        """

        println(io, "\n\nPossible fix, define:\n")
        println(io, possible_fix_definition)
    else
        println(io, "\n\n[WARN]: Non-standard rule layout found! Possible fix, define rule with the following arguments:\n")
        println(io, "rule.fform: ", error.fform)
        println(io, "rule.on: ", error.on)
        println(io, "rule.mnames: ", error.mnames)
        println(io, "rule.messages: ", error.messages)
        println(io, "rule.qnames: ", error.qnames)
        println(io, "rule.marginals: ", error.marginals)
        println(io, "rule.meta: ", error.meta)
    end
end

# Convert method into a markdown row
function convert_to_markdown(m::Method)
    node = get_node(m)
    inputs = get_inputs(m)
    output = ""
    for k in 1:length(inputs)
        output *= "| "
        k == 1 ? output *= node : output *= " | "
        output *= " $(inputs)[k]"
        output *= " |\n --- \n"
    end
    return output
end

# Extracts node from rule method
function get_node_from_rule_method(m::Method)
    _, decls, _, _ = Base.arg_decl_parts(m)
    return decls[2][2][8:(end - 1)]
end

# Extracts output from rule method
function get_output_from_rule_method(m::Method)
    _, decls, _, _ = Base.arg_decl_parts(m)
    return replace(decls[3][2], r"Type|Val|{|}|:|\(|\)|\,|Tuple|Int64" => "")
end

# Extracts name of message from rule method (e.g, :a, :out)
function get_message_names_from_rule_method(m::Method)
    _, decls, _, _ = Base.arg_decl_parts(m)
    return split(replace(decls[5][2], r"Type|Val|{|}|:|\(|\)|\," => ""))
end

# Extracts type of message from rule method (e.g., PointMass, NormalMeanVariance)
function get_message_types_from_rule_method(m::Method)
    _, decls, _, _ = Base.arg_decl_parts(m)
    tmp1 = replace(replace(decls[6][2][7:(end - 1)], r"ReactiveMP.ManyOf{<:Tuple{Vararg{" => ""), r"\, N}}}" => "xyz")
    tmp2 = strip.(strip.(split(tmp1, "Message")[2:end]), ',')
    tmp3 = map(x -> x == "xyz" ? "{<:ManyOf{<:Tuple{Vararg{Any, N}}}}" : x, tmp2)
    tmp4 = map(x -> x == r"xyz*" ? "{<:ManyOf{<:Tuple{Vararg{" * x[4:end] * ", N}}}}" : x, tmp3)
    tmp5 = map(x -> occursin("xyz", x) ? x[1:(end - 3)] : x, tmp4)
    return map(x -> isempty(x) ? "Any" : x, map(x -> x[4:(end - 1)], tmp5))
end

# Extracts messages from rule method (e.g., "μ(a) :: PointMass")
function get_messages_from_rule_method(m::Method)
    interfaces = get_message_names_from_rule_method(m)
    types = get_message_types_from_rule_method(m)
    return "μ(" .* interfaces .* ")" .* " :: " .* types
end

# Extracts name of marginal from rule method (e.g, :a, :out)
function get_marginal_names_from_rule_method(m::Method)
    _, decls, _, _ = Base.arg_decl_parts(m)
    return split(replace(decls[7][2], r"Type|Val|{|}|:|\(|\)|\," => ""))
end

# Extracts type of marginal from rule method (e.g., PointMass, NormalMeanVariance)
function get_marginal_types_from_rule_method(m::Method)
    _, decls, _, _ = Base.arg_decl_parts(m)
    tmp1 = replace(replace(decls[8][2][7:(end - 1)], r"ReactiveMP.ManyOf{<:Tuple{Vararg{" => ""), r"\, N}}}" => "xyz")
    tmp2 = strip.(strip.(split(tmp1, "Marginal")[2:end]), ',')
    tmp3 = map(x -> x == "xyz" ? "{<:ManyOf{<:Tuple{Vararg{Any, N}}}}" : x, tmp2)
    tmp4 = map(x -> x == r"xyz*" ? "{<:ManyOf{<:Tuple{Vararg{" * x[4:end] * ", N}}}}" : x, tmp3)
    tmp5 = map(x -> occursin("xyz", x) ? x[1:(end - 3)] : x, tmp4)
    return map(x -> isempty(x) ? "Any" : x, map(x -> x[4:(end - 1)], tmp5))
end

# Extracts marginals from rule method
function get_marginals_from_rule_method(m::Method)
    interfaces = get_marginal_names_from_rule_method(m)
    types = get_marginal_types_from_rule_method(m)
    return "q(" .* interfaces .* ")" .* " :: " .* types
end

# Extracts meta from rule method
function get_meta_from_rule_method(m::Method)
    _, decls, _, _ = Base.arg_decl_parts(m)
    return decls[9][2]
end

# Prints the rows corresponding to a single rule method in a table 
function print_rule_rows(m::Method)
    node = get_node_from_rule_method(m)
    output = get_output_from_rule_method(m)
    inputs = vcat(get_messages_from_rule_method(m), get_marginals_from_rule_method(m))
    meta = get_meta_from_rule_method(m)
    txt = ""
    for k in 1:length(inputs)
        txt *= "| "
        k == 1 ? txt *= node : nothing
        txt *= " | "
        k == 1 ? txt *= output : nothing
        txt *= " | "
        txt *= inputs[k]
        txt *= " | "
        k == 1 ? txt *= meta : nothing
        txt *= " |\n"
    end
    return txt
end

# Prints a table of all message passing update rules in ReactiveMP.
# Use `Markdown.parse` on the output of this function to get a prettified table.
function print_rules_table()
    mtds = methods(ReactiveMP.rule)
    """
                   | Node | Output | Inputs | Meta |
                   |:-----|:-------|:-------|:-----|
                   """ * mapreduce(ReactiveMP.print_rule_rows, *, mtds)
end
