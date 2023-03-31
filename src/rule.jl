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
call_rule_is_node_required(fformtype) = call_rule_is_node_required(as_node_functional_form(fformtype), fformtype)

call_rule_is_node_required(::ValidNodeFunctionalForm, fformtype) = CallRuleNodeNotRequired()
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
    @call_rule NodeType(:edge, Constraint) (argument1 = value1, argument2 = value2, ..., [ meta = ... ])

The `@call_rule` macro helps to call the `rule` method with an easier syntax. 
The structure of the macro is almost the same as in the `@rule` macro, but there is no `begin ... end` block, but instead each argument must have a specified value with the `=` operator.

See also: [`@rule`](@ref), [`rule`](@ref), [`@call_marginalrule`](@ref)
"""
macro call_rule(fform, args)
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

    m_names_arg, m_values_arg = call_rule_macro_parse_fn_args(inputs; specname = :messages, prefix = :m_, proxy = :(ReactiveMP.Message))
    q_names_arg, q_values_arg = call_rule_macro_parse_fn_args(inputs; specname = :marginals, prefix = :q_, proxy = :(ReactiveMP.Marginal))

    on_arg = call_rule_macro_construct_on_arg(on_type, on_index)

    distributionsym = gensym(:distributionsym)
    addonsym = gensym(:addonsym)

    output = quote
        # TODO: (bvdmitri At the moment we cannot really get the result of the addon by calling `@call_rule`
        $distributionsym, $addonsym = ReactiveMP.rule($fbottomtype, $on_arg, $(vconstraint)(), $m_names_arg, $m_values_arg, $q_names_arg, $q_values_arg, $meta, $addons, $node)
        $distributionsym
    end

    return esc(output)
end

"""
    Documentation placeholder
"""
macro test_rules(options, on, test_sequence)
    @capture(options, [option_entries__]) || error("Invalid options specification. Options should be in the form on an array.")

    with_float_conversions = false

    float64_atol  = 1e-12
    float32_atol  = 1e-6
    bigfloat_atol = 1e-12

    foreach(option_entries) do option_entry
        @capture(option_entry, (key_ = value_)) || error("Invalid option entry specification: $(option_entry). Option entry should be in the form of a 'key = value' pair.")
        if key === :with_float_conversions
            if value === :true
                with_float_conversions = true
            elseif value === :false
                with_float_conversions = false
            else
                error("Unknown value $(value) for option $(key). Can be either true or false.")
            end
        elseif key === :atol
            float64_atol  = float(value)
            float32_atol  = float(value)
            bigfloat_atol = float(value)
        elseif key === :float64_atol
            float64_atol = float(value)
        elseif key === :float32_atol
            float32_atol = float(value)
        elseif key === :bigfloat_atol
            bigfloat_atol = float(value)
        else
            error("Unknown option $(key)")
        end
    end

    @capture(test_sequence, [test_sequence_entries__]) || error("Invalid test sequence specification. Test sequence should be in the form of an array.")

    block      = Expr(:block)
    block.args = map(test_sequence_entries) do test_entry
        @capture(test_entry, (input = input_, output = output_)) || error("Invalid test entry specification: $(test_entry). Test entry should be in the form of a named tuple (input = ..., output = ...).")

        test_rule      = Expr(:block)
        test_output_s  = gensym()
        test_rule.args = [quote
            begin
                local $test_output_s = ReactiveMP.@call_rule($on, $input)
                @test ReactiveMP.custom_isapprox($test_output_s, $output; atol = $float64_atol)
                @test ReactiveMP.is_typeof_equal($test_output_s, $output)
            end
        end]

        if with_float_conversions
            @capture(input, (input_entries__,)) || error("Invalid input entries. Input entries should be in the form of a named tuple. ")

            # We filter out indices only for inputs that start with 'm_' or 'q_'
            # + we ignore `m_\q_* = nothing`
            inputs = map(first, filter(collect(enumerate(input_entries))) do i
                @capture(i[2], (key_ = value_))
                if key !== nothing && value !== :nothing
                    skey = string(key)
                    return startswith(skey, "m_") || startswith(skey, "q_")
                end
                return false
            end)

            function powerset(x::Vector{T}) where {T}
                result = Vector{T}[[]]
                for elem in x, j in eachindex(result)
                    push!(result, [result[j]; elem])
                end
                result
            end

            # Here we create all subsets of a input set, to modify their eltype
            indices_power_set = filter(!isempty, powerset(inputs))

            # We create a modified testset for Float32 inputs
            modified_f32_inputs = map(indices_power_set) do set
                cinput = deepcopy(input)
                for index in set
                    cinput.args[index].args[2] = MacroHelpers.expression_convert_eltype(Float32, cinput.args[index].args[2])
                end
                return (cinput, length(set) === length(inputs))
            end

            for m_f32_input in modified_f32_inputs
                m_f32_output = m_f32_input[2] ? MacroHelpers.expression_convert_eltype(Float32, output) : output
                output_s = gensym()
                push!(test_rule.args, quote
                    begin
                        local $output_s = ReactiveMP.@call_rule($on, $(m_f32_input[1]))
                        @test ReactiveMP.custom_isapprox($output_s, $m_f32_output; atol = $float32_atol)
                        @test ReactiveMP.is_typeof_equal($output_s, $m_f32_output)
                    end
                end)
            end

            # We create a modified testset for BigFloat inputs
            modified_bigf_inputs = map(indices_power_set) do set
                cinput = deepcopy(input)
                for index in set
                    cinput.args[index].args[2] = MacroHelpers.expression_convert_eltype(BigFloat, cinput.args[index].args[2])
                end
                return (cinput, true)
            end

            for m_bigf_input in modified_bigf_inputs
                m_bigf_output = m_bigf_input[2] ? MacroHelpers.expression_convert_eltype(BigFloat, output) : output
                output_dist = gensym()
                push!(test_rule.args, quote
                    begin
                        local $output_dist = ReactiveMP.@call_rule($on, $(m_bigf_input[1]))
                        @test ReactiveMP.custom_isapprox($output_dist, $m_bigf_output; atol = $bigfloat_atol)
                        @test ReactiveMP.is_typeof_equal($output_dist, $m_bigf_output)
                    end
                end)
            end
        end

        return test_rule
    end

    return esc(block)
end

macro test_rules(on, test_sequence)
    return :(@test_rules [] $on $test_sequence)
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
    return (out = m_out, μ = prod(ProdAnalytical(), NormalMeanPrecision(mean(m_out), mean(q_τ)), m_μ))
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

    m_names_arg, m_values_arg = call_rule_macro_parse_fn_args(inputs; specname = :messages, prefix = :m_, proxy = :(ReactiveMP.Message))
    q_names_arg, q_values_arg = call_rule_macro_parse_fn_args(inputs; specname = :marginals, prefix = :q_, proxy = :(ReactiveMP.Marginal))

    on_arg = call_rule_macro_construct_on_arg(on_type, on_index)

    output = quote
        ReactiveMP.marginalrule($fbottomtype, $on_arg, $m_names_arg, $m_values_arg, $q_names_arg, $q_values_arg, $meta, $node)
    end

    return esc(output)
end

"""
    Documentation placeholder
"""
macro test_marginalrules(options, on, test_sequence)
    @capture(options, [option_entries__]) || error("Invalid options specification. Options should be in the form on an array.")

    with_float_conversions = false

    float64_atol  = 1e-12
    float32_atol  = 1e-6
    bigfloat_atol = 1e-12

    foreach(option_entries) do option_entry
        @capture(option_entry, (key_ = value_)) || error("Invalid option entry specification: $(option_entry). Option entry should be in the form of a 'key = value' pair.")
        if key === :with_float_conversions
            if value === :true
                with_float_conversions = true
            elseif value === :false
                with_float_conversions = false
            else
                error("Unknown value $(value) for option $(key). Can be either true or false.")
            end
        elseif key === :atol
            float64_atol  = float(value)
            float32_atol  = float(value)
            bigfloat_atol = float(value)
        elseif key === :float64_atol
            float64_atol = float(value)
        elseif key === :float32_atol
            float32_atol = float(value)
        elseif key === :bigfloat_atol
            bigfloat_atol = float(value)
        else
            error("Unknown option $(key)")
        end
    end

    @capture(test_sequence, [test_sequence_entries__]) || error("Invalid test sequence specification. Test sequence should be in the form of an array.")

    block      = Expr(:block)
    block.args = map(test_sequence_entries) do test_entry
        @capture(test_entry, (input = input_, output = output_)) || error("Invalid test entry specification: $(test_entry). Test entry should be in the form of a named tuple (input = ..., output = ...).")

        test_rule      = Expr(:block)
        test_output_s  = gensym()
        test_rule.args = [quote
            begin
                local $test_output_s = ReactiveMP.@call_marginalrule($on, $input)
                @test ReactiveMP.custom_isapprox($test_output_s, $output; atol = $float64_atol)
                @test ReactiveMP.is_typeof_equal($test_output_s, $output)
            end
        end]

        if with_float_conversions
            @capture(input, (input_entries__,)) || error("Invalid input entries. Input entries should be in the form of a named tuple. ")

            # We filter out indices only for inputs that start with 'm_' or 'q_'
            inputs = map(first, filter(collect(enumerate(input_entries))) do i
                @capture(i[2], (key_ = value_))
                if key !== nothing
                    skey = string(key)
                    return startswith(skey, "m_") || startswith(skey, "q_")
                end
                return false
            end)

            function powerset(x::Vector{T}) where {T}
                result = Vector{T}[[]]
                for elem in x, j in eachindex(result)
                    push!(result, [result[j]; elem])
                end
                result
            end

            # Here we create all subsets of a input set, to modify their eltype
            indices_power_set = filter(!isempty, powerset(inputs))

            # We create a modified testset for Float32 inputs
            modified_f32_inputs = map(indices_power_set) do set
                cinput = deepcopy(input)
                for index in set
                    cinput.args[index].args[2] = MacroHelpers.expression_convert_eltype(Float32, cinput.args[index].args[2])
                end
                return (cinput, length(set) === length(inputs))
            end

            for m_f32_input in modified_f32_inputs
                m_f32_output = m_f32_input[2] ? MacroHelpers.expression_convert_eltype(Float32, output) : output
                output_dist = gensym()
                push!(test_rule.args, quote
                    begin
                        local $output_dist = ReactiveMP.@call_marginalrule($on, $(m_f32_input[1]))
                        @test ReactiveMP.custom_isapprox($output_dist, $m_f32_output; atol = $float32_atol)
                        # @test ReactiveMP.is_typeof_equal($output_s, $m_f32_output) # broken
                    end
                end)
            end

            # We create a modified testset for BigFloat inputs
            modified_bigf_inputs = map(indices_power_set) do set
                cinput = deepcopy(input)
                for index in set
                    cinput.args[index].args[2] = MacroHelpers.expression_convert_eltype(BigFloat, cinput.args[index].args[2])
                end
                return (cinput, true)
            end

            for m_bigf_input in modified_bigf_inputs
                m_bigf_output = m_bigf_input[2] ? MacroHelpers.expression_convert_eltype(BigFloat, output) : output
                output_dist = gensym()
                push!(test_rule.args, quote
                    begin
                        local $output_dist = ReactiveMP.@call_marginalrule($on, $(m_bigf_input[1]))
                        @test ReactiveMP.custom_isapprox($output_dist, $m_bigf_output; atol = $bigfloat_atol)
                        # @test ReactiveMP.is_typeof_equal($output_s, $m_bigf_output) # broken
                    end
                end)
            end
        end

        return test_rule
    end

    return esc(block)
end

macro test_marginalrules(on, test_sequence)
    return :(@test_marginalrules [] $on $test_sequence)
end

# Errors 

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

function interfaces(stub::NodeErrorStub)
    return fill(nothing, stub.counter)
end

rule_method_error_extract_fform(f::Function) = string("typeof(", f, ")")
rule_method_error_extract_fform(f)           = string(f)

rule_method_error_extract_on(::Val{T}) where {T}              = string(":", T)
rule_method_error_extract_on(::Tuple{Val{T}, Int}) where {T}  = string("(", rule_method_error_extract_on(Val{T}()), ", k)")
rule_method_error_extract_on(::Tuple{Val{T}, N}) where {T, N} = string("(", rule_method_error_extract_on(Val{T}()), ", ", convert(Int, N), ")")

rule_method_error_extract_vconstraint(something) = typeof(something)

rule_method_error_extract_names(::Val{T}) where {T} = map(sT -> __extract_val_type(split_underscored_symbol(Val{sT}())), T)
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

rule(fform, on, vconstraint, mnames, messages, qnames, marginals, meta, addons, __node) =
    throw(RuleMethodError(fform, on, vconstraint, mnames, messages, qnames, marginals, meta, addons, __node))

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

        spec = Vector(undef, 4length(interfaces(node)))

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

    spec = Vector(undef, 4length(interfaces(node)))

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
