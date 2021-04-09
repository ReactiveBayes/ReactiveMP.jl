export rule, marginalrule
export @rule, @marginalrule

using MacroTools

"""
    rule(fform, on, vconstraint, mnames, messages, qnames, marginals, meta, __node)

This function is used to compute an outbound message for a given node

# Arguments

- `fform`: Functional form of the node in form of a type of the node, e.g. `::Type{ <: NormalMeanVariance }` or `::typeof(+)`
- `on`: Outbound interface's tag for which a message has to be computed, e.g. `::Type{ Val{:μ} }` or `::Type{ Val{:μ} }`
- `vconstraint`: Variable constraints for an outbound interface, e.g. `Marginalisation` or `MomentMatching`
- `mnames`: Ordered messages names in form of the Val type, eg. ::Type{ Val{ (:mean, :precision) } }
- `messages`: Tuple of message of the same length as `mnames` used to compute an outbound message
- `qnames`: Ordered marginal names in form of the Val type, eg. ::Type{ Val{ (:mean, :precision) } }
- `marginals`: Tuple of marginals of the same length as `qnames` used to compute an outbound message
- `meta`: Extra meta information
- `__node`: Node reference

See also: [`@rule`](@ref), [`marginalrule`], [`@marginalrule`](@ref)
"""
function rule end

"""
    marginalrule(fform, on, mnames, messages, qnames, marginals, meta, __node)

This function is used to compute a local joint marginal for a given node

# Arguments

- `fform`: Functional form of the node in form of a type of the node, e.g. `::Type{ <: NormalMeanVariance }` or `::typeof(+)`
- `on`: Local joint marginal tag , e.g. `::Type{ Val{ :mean_precision } }` or `::Type{ Val{ :out_mean_precision } }`
- `mnames`: Ordered messages names in form of the Val type, eg. ::Type{ Val{ (:mean, :precision) } }
- `messages`: Tuple of message of the same length as `mnames` used to compute an outbound message
- `qnames`: Ordered marginal names in form of the Val type, eg. ::Type{ Val{ (:mean, :precision) } }
- `marginals`: Tuple of marginals of the same length as `qnames` used to compute an outbound message
- `meta`: Extra meta information
- `__node`: Node reference

See also: [`rule`], [`@rule`](@ref) [`@marginalrule`](@ref)
"""
function marginalrule end

# Macro code

function rule_macro_parse_on_tag(on)
    if @capture(on, :name_)
        return :(Type{ Val{ $(QuoteNode(name)) } }), nothing, nothing
    elseif @capture(on, (:name_, index_))
        return :(Tuple{ Val{$(QuoteNode(name))}, Int}), index, :($index = on[2])
    else
        error("Error in macro. on specification is incorrect")
    end
end

function rule_macro_parse_fn_args(inputs; specname, prefix, proxy)
    finputs = filter((i) -> startswith(string(first(i)), string(prefix)), inputs)

    names  = map(first, finputs)
    types  = map((i) -> MacroHelpers.proxy_type(proxy, last(i)), finputs)

    @assert all((n) -> length(string(n)) > 2, names)  || error("Empty $(specname) name found in arguments")

    init_block = map(enumerate(names)) do (index, iname)
        return :($(iname) = getdata($(specname)[$(index)]))
    end

    out_names = length(names) === 0 ? :Nothing : :(Type{ Val{ $(tuple(map(n -> Symbol(string(n)[(length(string(prefix)) + 1):end]), names)...)) } })
    out_types = length(types) === 0 ? :Nothing : :(Tuple{ $(types...) })

    return out_names, out_types, init_block
end

function rule_function_expression(body::Function, fuppertype, on_type, vconstraint, m_names, m_types, q_names, q_types, metatype, whereargs)
    return quote
        function ReactiveMP.rule(
            fform           :: $(fuppertype),
            on              :: $(on_type),
            vconstraint     :: $(vconstraint),
            messages_names  :: $(m_names),
            messages        :: $(m_types),
            marginals_names :: $(q_names),
            marginals       :: $(q_types),
            meta            :: $(metatype),
            __node
        ) where { $(whereargs...) }
            $(body())
        end
    end
end

function marginalrule_function_expression(body::Function, fuppertype, on_type, m_names, m_types, q_names, q_types, metatype, whereargs)
    return quote
        function ReactiveMP.marginalrule(
            fform           :: $(fuppertype),
            on              :: $(on_type),
            messages_names  :: $(m_names),
            messages        :: $(m_types),
            marginals_names :: $(q_names),
            marginals       :: $(q_types),
            meta            :: $(metatype),
            __node
        ) where { $(whereargs...) }
            $(body())
        end
    end
end



import .MacroHelpers

"""
    Documentation placeholder
"""
macro rule(fform, lambda)
    @capture(fform, fformtype_(on_, vconstraint_, options__)) || 
        error("Error in macro. Functional form specification should in the form of 'fformtype_(on_, vconstraint_, options__)'")

    @capture(lambda, (args_ where { whereargs__ } = body_) | (args_ = body_)) || 
        error("Error in macro. Lambda body specification is incorrect")

    @capture(args, (inputs__, meta::metatype_) | (inputs__, )) || 
        error("Error in macro. Lambda body arguments specification is incorrect")

    fuppertype                       = MacroHelpers.upper_type(fformtype)
    on_type, on_index, on_index_init = rule_macro_parse_on_tag(on)
    whereargs                        = whereargs === nothing ? [] : whereargs
    metatype                         = metatype === nothing ? :Any : metatype
    
    options = map(options) do option
        @capture(option, name_ = value_) || error("Error in macro. Option specification '$(option)' is incorrect")x
        return (name, value)
    end

    inputs = map(inputs) do input
        @capture(input, iname_::itype_) || error("Error in macro. Input $(input) is incorrect")
        return (iname, itype)
    end

    m_names, m_types, m_init_block = rule_macro_parse_fn_args(inputs, specname = :messages, prefix = :m_, proxy = :Message)
    q_names, q_types, q_init_block = rule_macro_parse_fn_args(inputs, specname = :marginals, prefix = :q_, proxy = :Marginal)

    output = quote
        $(
            rule_function_expression(fuppertype, on_type, vconstraint, m_names, m_types, q_names, q_types, metatype, whereargs) do
                return quote
                    $(on_index_init)
                    $(m_init_block...)
                    $(q_init_block...)
                    $(body)
                end
            end
        )
    end

    foreach(options) do option 

        # Symmetrical option
        if first(option) === :symmetrical
            @capture(last(option), [ names__ ]) || 
                error("Invalid symmetrical names specification. Name should be a symbol.")

            @assert length(names) > 1 "Invalid symmetrical names specification. Length of names should be greater than 1."

            names = map(names) do name
                @capture(name, :s_) || error("Invalid symmetrical name specification. Name should be a symbol.")
                return s
            end

            indices = map(names) do name
                return findfirst(input -> isequal(string("m_", name), string(first(input))) || isequal(string("q_", name), string(first(input))), inputs)
            end

            foreach(enumerate(indices)) do (i, index)
                @assert index !== nothing "Name $(names[i]) does not exist in arguments list"
            end

            prefixes = map(index -> string(first(inputs[index]))[1:2], indices)

            @assert length(Set(prefixes)) === 1 "It is not possible to mix symmetric arguments from messages and marginals"

            prefix = first(prefixes)
            swaps  = Iterators.flatten(map(i -> map(j -> (indices[i], indices[j]), i+1:length(indices)), 1:length(indices)))

            is_messages  = isequal(prefix, "m_")
            is_marginals = isequal(prefix, "q_")

            foreach(swaps) do swap 

                messages  = is_messages ? MacroHelpers.rearranged_tuple(:messages, length(m_init_block), swap) : :(messages)
                marginals = is_marginals ? MacroHelpers.rearranged_tuple(:marginals, length(m_init_block), swap) : :(marginals)

                swapped_m_types = is_messages ? :(Tuple{$(swapped(m_types.args[2:end], first(swap), last(swap))...)}) : m_types
                swapped_q_types = is_marginals ? :(Tuple{$(swapped(q_types.args[2:end], first(swap), last(swap))...)}) : q_types

                @assert !is_messages || swapped_m_types != m_types "Message types are the same after arguments swap for indices = $(swap)"
                @assert !is_marginals || swapped_q_types != q_types "Marginal types are the same after arguments swap for indices = $(swap)"

                output = quote
                    $output
                    $(
                        rule_function_expression(fuppertype, on_type, vconstraint, m_names, swapped_m_types, q_names, swapped_q_types, metatype, whereargs) do
                            return :(ReactiveMP.rule(fform, on, vconstraint, messages_names, $(messages), marginals_names, $(marginals), meta, __node))
                        end 
                    )
                end
            end
        else 
            error("Unknown option: $(first(option)) in rule specification")
        end
        
    end

    return esc(output)
end

macro call_rule(fform, args)
    @capture(fform, fformtype_(on_, vconstraint_)) || 
        error("Error in macro. Functional form specification should in the form of 'fformtype_(on_, vconstraint_)'")

    @capture(args, (inputs__, meta = meta_) | (inputs__, )) || 
        error("Error in macro. Arguments specification is incorrect")

    fuppertype                       = MacroHelpers.upper_type(fformtype)
    fbottomtype                      = MacroHelpers.bottom_type(fformtype)
    on_type, on_index, on_index_init = rule_macro_parse_on_tag(on)

    inputs = map(inputs) do input
        @capture(input, iname_ = ivalue_) || error("Error in macro. Argument $(input) is incorrect")
        return (iname, ivalue)
    end

    messages  = map(i -> (Symbol(string(first(i))[3:end]), last(i)), filter(i -> startswith(string(first(i)), "m_"), inputs))
    marginals = map(i -> (Symbol(string(first(i))[3:end]), last(i)), filter(i -> startswith(string(first(i)), "q_"), inputs))

    foreach((message) -> begin @assert length(string(first(message))) !== 0 "Empty message argument name" end, messages)
    foreach((marginal) -> begin @assert length(string(first(marginal))) !== 0 "Empty marginal argument name" end, marginals)

    m_names, m_values = tuple(first.(messages)...), tuple(last.(messages)...)
    q_names, q_values = tuple(first.(marginals)...), tuple(last.(marginals)...)

    m_names_arg  = isempty(m_names) ? :nothing : :(Val{ $(m_names) })
    m_values_arg = isempty(m_names) ? :nothing : :($(map(m_value -> :(as_message($m_value)), m_values)...), )
    q_names_arg  = isempty(q_names) ? :nothing : :(Val{ $(q_names) })
    q_values_arg = isempty(q_names) ? :nothing : :($(map(q_value -> :(as_marginal($q_value)), q_values)...), )

    on_arg = MacroHelpers.bottom_type(on_type)

    output = quote
        rule($fbottomtype, $on_arg, $(vconstraint)(), $m_names_arg, $m_values_arg, $q_names_arg, $q_values_arg, $meta, nothing)
    end

    return esc(output)
end

"""
    Documentation placeholder
"""
macro marginalrule(fform, lambda)
    @capture(fform, fformtype_(on_)) || 
        error("Error in macro. Functional form specification should in the form of 'fformtype_(on_)'")

    @capture(lambda, (args_ where { whereargs__ } = body_) | (args_ = body_)) || 
        error("Error in macro. Lambda body specification is incorrect")

    @capture(args, (inputs__, meta::metatype_) | (inputs__, )) || 
        error("Error in macro. Lambda body arguments speicifcation is incorrect")

    fuppertype                       = MacroHelpers.upper_type(fformtype)
    on_type, on_index, on_index_init = rule_macro_parse_on_tag(on)
    whereargs                        = whereargs === nothing ? [] : whereargs
    metatype                         = metatype === nothing ? :Any : metatype

    inputs = map(inputs) do input
        @capture(input, iname_::itype_) || error("Error in macro. Input $(input) is incorrect")
        return (iname, itype)
    end

    m_names, m_types, m_init_block = rule_macro_parse_fn_args(inputs, specname = :messages, prefix = :m_, proxy = :Message)
    q_names, q_types, q_init_block = rule_macro_parse_fn_args(inputs, specname = :marginals, prefix = :q_, proxy = :Marginal)

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