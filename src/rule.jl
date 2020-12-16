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

"""
    Documentation placeholder
"""
macro rule(fform, lambda)
    @capture(fform, fformtype_(on_, vconstraint_)) || error("Error in macro. Functional form specification should in the form of 'fformtype_(on_, vconstraint_)'")

    on_type, on_where_Ts = __extract_on_args_macro_rule(on)

    @capture(lambda, (args_ where { whereargs__ } = body_) | (args_ = body_)) || error("Error in macro. Lambda body specification is incorrect")
    @capture(args, (inputs__, meta::metatype_) | (inputs__, )) || error("Error in macro. Lambda body arguments speicifcation is incorrect")

    whereargs = whereargs === nothing ? [] : whereargs

    inputs = map(inputs) do input
        @capture(input, iname_::itype_) || error("Error in macro. Input $(input) is incorrect")
        return (iname, itype)
    end

    m_names, m_types, m_init_block = __extract_fn_args_macro_rule(inputs, specname = :messages, prefix = :m_, proxytype = :Message)
    q_names, q_types, q_init_block = __extract_fn_args_macro_rule(inputs, specname = :marginals, prefix = :q_, proxytype = :Marginal)

    metatype = metatype === nothing ? :Nothing : metatype

    output = quote
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
        ) where { $(whereargs...), $(on_where_Ts...) }
            $(m_init_block...)
            $(q_init_block...)
            $(body)
        end
    end

    return esc(output)
end

"""
    Documentation placeholder
"""
macro marginalrule(fform, lambda)

    @capture(fform, fformtype_(on_)) || error("Error in macro. Functional form specification should in the form of 'fformtype_(on_)'")

    on_type, on_where_Ts = __extract_on_args_macro_rule(on)

    @capture(lambda, (args_ where { whereargs__ } = body_) | (args_ = body_)) || error("Error in macro. Lambda body specification is incorrect")
    @capture(args, (inputs__, meta::metatype_) | (inputs__, )) || error("Error in macro. Lambda body arguments speicifcation is incorrect")

    whereargs = whereargs === nothing ? [] : whereargs

    inputs = map(inputs) do input
        @capture(input, iname_::itype_) || error("Error in macro. Input $(input) is incorrect")
        return (iname, itype)
    end

    m_names, m_types, m_init_block = __extract_fn_args_macro_rule(inputs, specname = :messages, prefix = :m_, proxytype = :Message)
    q_names, q_types, q_init_block = __extract_fn_args_macro_rule(inputs, specname = :marginals, prefix = :q_, proxytype = :Marginal)

    metatype = metatype === nothing ? :Nothing : metatype

    output = quote
        function ReactiveMP.marginalrule(
            fform           :: $(fformtype),
            on              :: $(on_type),
            messages_names  :: $(m_names),
            messages        :: $(m_types),
            marginals_names :: $(q_names),
            marginals       :: $(q_types),
            meta            :: $(metatype),
            __node
        ) where { $(whereargs...), $(on_where_Ts...) }
            $(m_init_block...)
            $(q_init_block...)
            $(body)
        end
    end
    
    return esc(output)
end