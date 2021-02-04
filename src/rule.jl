export rule, marginalrule
export @rule, @marginalrule

"""
    Documentation placeholder
"""
function rule end

"""
    Documentation placeholder
"""
function marginalrule end


function __write_rule_output(body::Function, fformtype, on_type, vconstraint, m_names, m_types, q_names, q_types, metatype, whereargs)
    return quote
        function ReactiveMP.rule(
            fform           :: $(fformtype),
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

import .MacroHelpers

"""
    Documentation placeholder
"""
macro rule(fform, lambda)
    @capture(fform, fformtype_(on_, vconstraint_, options__)) || error("Error in macro. Functional form specification should in the form of 'fformtype_(on_, vconstraint_, options__)'")

    options = map(options) do option
        @capture(option, name_ = value_)
        return (name, value)
    end

    fuppertype = MacroHelpers.upper_type(fformtype)
    on_type, on_index = __extract_on_args_macro_rule(on)

    on_index_init = on_index === nothing ? :(nothing) : :($on_index = on[2])

    @capture(lambda, (args_ where { whereargs__ } = body_) | (args_ = body_)) || error("Error in macro. Lambda body specification is incorrect")
    @capture(args, (inputs__, meta::metatype_) | (inputs__, )) || error("Error in macro. Lambda body arguments speicifcation is incorrect")

    whereargs = whereargs === nothing ? [] : whereargs

    inputs = map(inputs) do input
        @capture(input, iname_::itype_) || error("Error in macro. Input $(input) is incorrect")
        return (iname, itype)
    end

    m_names, m_types, m_init_block = __extract_fn_args_macro_rule(inputs, specname = :messages, prefix = :m_, proxy = :Message)
    q_names, q_types, q_init_block = __extract_fn_args_macro_rule(inputs, specname = :marginals, prefix = :q_, proxy = :Marginal)

    metatype = metatype === nothing ? :Nothing : metatype

    output = quote
        $(
            __write_rule_output(fuppertype, on_type, vconstraint, m_names, m_types, q_names, q_types, metatype, whereargs) do
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
            @capture(last(option), [ names__ ]) || error("Invalid symmetrical names specification. Name should be a symbol.")

            @assert length(names) > 1 "Invalid symmetrical names specification. Length of names should be grater than 1."

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

                messages  = is_messages ? __rearrange_tupled_arguments(:messages, length(m_init_block), swap) : :(messages)
                marginals = is_marginals ? __rearrange_tupled_arguments(:marginals, length(m_init_block), swap) : :(marginals)

                swapped_m_types = is_messages ? :(Tuple{$(swap_indices_array(m_types.args[2:end], first(swap), last(swap))...)}) : m_types
                swapped_q_types = is_marginals ? :(Tuple{$(swap_indices_array(q_types.args[2:end], first(swap), last(swap))...)}) : q_types

                @assert !is_messages || swapped_m_types != m_types "Message types are the same after arguments swap for indices = $(swap)"
                @assert !is_marginals || swapped_q_types != q_types "Marginal types are the same after arguments swap for indices = $(swap)"

                output = quote
                    $output
                    $(
                        __write_rule_output(fuppertype, on_type, vconstraint, m_names, swapped_m_types, q_names, swapped_q_types, metatype, whereargs) do
                            return quote
                                return ReactiveMP.rule(fform, on, vconstraint, messages_names, $(messages), marginal_names, $(marginals), meta, __node)
                            end
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

"""
    Documentation placeholder
"""
macro marginalrule(fform, lambda)

    @capture(fform, fformtype_(on_)) || error("Error in macro. Functional form specification should in the form of 'fformtype_(on_)'")

    fuppertype = MacroHelpers.upper_type(fformtype)
    on_type, on_index = __extract_on_args_macro_rule(on)

    on_index_init = on_index === nothing ? :(nothing) : :($on_index = on[2])

    @capture(lambda, (args_ where { whereargs__ } = body_) | (args_ = body_)) || error("Error in macro. Lambda body specification is incorrect")
    @capture(args, (inputs__, meta::metatype_) | (inputs__, )) || error("Error in macro. Lambda body arguments speicifcation is incorrect")

    whereargs = whereargs === nothing ? [] : whereargs

    inputs = map(inputs) do input
        @capture(input, iname_::itype_) || error("Error in macro. Input $(input) is incorrect")
        return (iname, itype)
    end

    m_names, m_types, m_init_block = __extract_fn_args_macro_rule(inputs, specname = :messages, prefix = :m_, proxy = :Message)
    q_names, q_types, q_init_block = __extract_fn_args_macro_rule(inputs, specname = :marginals, prefix = :q_, proxy = :Marginal)

    metatype = metatype === nothing ? :Nothing : metatype

    output = quote
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
            $(on_index_init)
            $(m_init_block...)
            $(q_init_block...)
            $(body)
        end
    end
    
    return esc(output)
end