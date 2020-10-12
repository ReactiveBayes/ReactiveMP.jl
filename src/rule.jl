export @rule

using MacroTools

function __extract_fform_macro_rule(fform)
    @capture(fform, (form => form_)) || 
        error("Error in macro: functional form of rule should have a (form => Type{ <: Distribution } or typeof(fn)) signature")
    return form
end

function __extract_on_macro_rule(on)
    @capture(on, (on => :name_)) ||
        error("Error in macro: on specification of rule should have a (on => :name) signature")
    return :(Type{ Val{ $(QuoteNode(name)) } })
end

function __extract_vconstraint_macro_rule(vconstraint)
    vconstraint !== :Nothing || return vconstraint
    @capture(vconstraint, (vconstraint => Constraint_)) ||
        error("Error in macro: edge specification of rule should have a (vconstraint => Constraint) signature")
    return Constraint
end

function __extract_fn_args_macro_rule(fn_specification; specname, prefix, proxytype)
    if @capture(fn_specification, (name_ => Nothing)) && name === specname
        return :Nothing, :Nothing, [], []
    end
    
    @capture(fn_specification, (name_ => args_specification_)) && name === specname ||
        error("Error in macro: $(specname) specification of rule should have a ($(specname) => (in1::Type1, in2::Type2) [ where ... ]) or Nothing signature") 
    
    where_Ts = []
    
    while @capture(args_specification, next_args_specification_ where { Ts__ })
        append!(where_Ts, Ts)
        args_specification = next_args_specification
    end
    
    @capture(args_specification, (entries__, )) ||
        error("Error in macro: messages specification of rule should have a ($(specname) => (in1::Type1, in2::Type2) [ where ... ]) or Nothing signature") 
    
    length(entries) !== 0 || error("Error in macro: $(specname) length should be greater than zero")
    
    specs = map(entries) do entry
        @capture(entry, name_::Type_) || 
            error("Error in macro: messages specification of rule should have a ($(specname) => (in1::Type1, in2::Type2) [ where ... ]) or Nothing signature") 
        return (name, Type)
    end

    all(d -> startswith(string(d[1]), string(prefix)), specs) || error("Error in macro: All names in $(specname) should be prefixed with $(prefix)")
    all(d -> length(string(d[1])) > 2, specs) || error("Error in macro: Empty name in $(specname) specification")

    args_names     = specs !== nothing ? map(a -> begin @views Symbol(string(a[1])[3:end]) end, specs) : nothing
    args_names_val = args_names !== nothing ? :(Type{ Val{ $(tuple(args_names...)) } }) : (:(Nothing))
    args_types     = specs !== nothing ? map(a -> :($(proxytype){ <: $(a[2]) }), specs) : nothing
    args_types_val = args_types !== nothing ? :(Tuple{ $(args_types...) }) : (:(Nothing))
    init_block     = args_names !== nothing ? map(i_name -> :($(Symbol(prefix, i_name[2])) = $(specname)[$(i_name[1])]), enumerate(args_names)) : [ :nothing ]
    
    return args_names_val, args_types_val, init_block, where_Ts
end

function __extract_meta_macro_rule(meta::Expr)
    @capture(meta, (meta => Meta_)) ||
        error("Invalid rule macro call: meta specification of rule should have a (meta => Type) signature")
    return Meta
end

macro rule(fform, on, vconstraint, messages, marginals, meta, fn)
    
    m_names, m_types, m_init_block, m_where_Ts = __extract_fn_args_macro_rule(messages; specname = :messages, prefix = :m_, proxytype = :Message)
    q_names, q_types, q_init_block, q_where_Ts = __extract_fn_args_macro_rule(marginals; specname = :marginals, prefix = :q_, proxytype = :Marginal)
    
    result = quote
        function ReactiveMP.rule(
            fform           :: $(__extract_fform_macro_rule(fform)),
            on              :: $(__extract_on_macro_rule(on)),
            vconstraint     :: $(__extract_vconstraint_macro_rule(vconstraint)),
            messages_names  :: $(m_names),
            messages        :: $(m_types),
            marginals_names :: $(q_names),
            marginals       :: $(q_types),
            meta            :: $(__extract_meta_macro_rule(meta)),
            __node
        ) where { $(m_where_Ts...), $(q_where_Ts...) }
            $(m_init_block...)
            $(q_init_block...)
            $(fn)
        end
    end
    
    return esc(result)
end

macro marginalrule(fform, on, messages, marginals, meta, fn)
    
    m_names, m_types, m_init_block, m_where_Ts = __extract_fn_args_macro_rule(messages; specname = :messages, prefix = :m_, proxytype = :Message)
    q_names, q_types, q_init_block, q_where_Ts = __extract_fn_args_macro_rule(marginals; specname = :marginals, prefix = :q_, proxytype = :Marginal)
    
    result = quote
        function ReactiveMP.marginalrule(
            fform           :: $(__extract_fform_macro_rule(fform)),
            on              :: $(__extract_on_macro_rule(on)),
            messages_names  :: $(m_names),
            messages        :: $(m_types),
            marginals_names :: $(q_names),
            marginals       :: $(q_types),
            meta            :: $(__extract_meta_macro_rule(meta)),
            __node
        ) where { $(m_where_Ts...), $(q_where_Ts...) }
            $(m_init_block...)
            $(q_init_block...)
            $(fn)
        end
    end
    
    return esc(result)
end