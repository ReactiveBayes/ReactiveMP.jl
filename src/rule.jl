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
macro rule(fformtype, on, vconstraint, messages, marginals, meta, fn)
    
    m_names, m_types, m_init_block, m_where_Ts = __extract_fn_args_macro_rule(messages; specname = :messages, prefix = :m_, proxytype = :Message)
    q_names, q_types, q_init_block, q_where_Ts = __extract_fn_args_macro_rule(marginals; specname = :marginals, prefix = :q_, proxytype = :Marginal)
    
    on_type, on_where_Ts = __extract_on_macro_rule(on)

    result = quote
        function ReactiveMP.rule(
            fform           :: $(__extract_fformtype_macro_rule(fformtype)),
            on              :: $(on_type),
            vconstraint     :: $(__extract_vconstraint_macro_rule(vconstraint)),
            messages_names  :: $(m_names),
            messages        :: $(m_types),
            marginals_names :: $(q_names),
            marginals       :: $(q_types),
            meta            :: $(__extract_meta_macro_rule(meta)),
            __node
        ) where { $(m_where_Ts...), $(q_where_Ts...), $(on_where_Ts...) }
            $(m_init_block...)
            $(q_init_block...)
            $(fn)
        end
    end
    
    return esc(result)
end

"""
    Documentation placeholder
"""
macro marginalrule(fformtype, on, messages, marginals, meta, fn)
    
    m_names, m_types, m_init_block, m_where_Ts = __extract_fn_args_macro_rule(messages; specname = :messages, prefix = :m_, proxytype = :Message)
    q_names, q_types, q_init_block, q_where_Ts = __extract_fn_args_macro_rule(marginals; specname = :marginals, prefix = :q_, proxytype = :Marginal)
    
    on_type, on_where_Ts = __extract_on_macro_rule(on)

    result = quote
        function ReactiveMP.marginalrule(
            fform           :: $(__extract_fformtype_macro_rule(fformtype)),
            on              :: $(on_type),
            messages_names  :: $(m_names),
            messages        :: $(m_types),
            marginals_names :: $(q_names),
            marginals       :: $(q_types),
            meta            :: $(__extract_meta_macro_rule(meta)),
            __node
        ) where { $(m_where_Ts...), $(q_where_Ts...), $(on_where_Ts...) }
            $(m_init_block...)
            $(q_init_block...)
            $(fn)
        end
    end
    
    return esc(result)
end