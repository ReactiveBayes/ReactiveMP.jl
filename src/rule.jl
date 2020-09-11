export @rule

using MacroTools

function __extract_fform_macro_rule(fform)
    @capture(fform, (form => form_)) || 
        error("Invalid rule macro call: functional form of rule should have a (form => Type{ <: Distribution } or typeof(fn)) signature")
    return form
end

function __extract_on_macro_rule(on)
    @capture(on, (on => :name_)) ||
        error("Invalid rule macro call: on specification of rule should have a (on => :name) signature")
    return :(Type{ Val{ $(QuoteNode(name)) } })
end

function __extract_vconstraint_macro_rule(vconstraint)
    vconstraint !== :Nothing || return vconstraint
    @capture(vconstraint, (vconstraint => Constraint_)) ||
        error("Invalid rule macro call: edge specification of rule should have a (vconstraint => Constraint) signature")
    return Constraint
end

function __extract_messages_macro_rule(messages)
    if @capture(messages, (messages => Nothing))
        return [], nothing
    end
    
    @capture(messages, (messages => msgs_spec_)) ||
        error("Invalid rule macro call: messages specification of rule should have a (messages => (in1::Dirac{T}, in2::Dirac{T}) [ where ... ]) or Nothing signature") 
    
    where_Ts = []
    
    while @capture(msgs_spec, next_msgs_spec_ where { Ts__ })
        append!(where_Ts, Ts)
        msgs_spec = next_msgs_spec
    end
    
    @capture(msgs_spec, (entries__, )) ||
        error("Invalid rule macro call: messages specification of rule should have a (messages => (in1::Dirac{T}, in2::Dirac{T}) [ where ... ]) or Nothing signature") 
    
    length(entries) !== 0 || error("Invalid rule macro call: messages length should be greater than zero")
    
    specs = map(entries) do entry
        @capture(entry, name_::Type_) || 
            error("Invalid rule macro call: messages specification of rule should have a (messages => (in1::Dirac{T}, in2::Dirac{T}) [ where ... ]) or Nothing signature") 
        return (name, Type)
    end
    
    all(d -> startswith(string(d[1]), "m_"), specs) || error("Invalid rule macro call: All names in messages should be prefixed with m_")
    all(d -> length(string(d[1])) > 2, specs) || error("Invalid rule macro call: Empty name in messages specification")
    
    return where_Ts, specs
    
end

function __extract_marginals_macro_rule(marginals::Expr)
    if @capture(marginals, (marginals => Nothing))
        return [], nothing
    end
    
    @capture(marginals, (marginals => msgs_spec_)) ||
        error("Invalid rule macro call: marginals specification of rule should have a (marginals => (in1::Dirac{T}, in2::Dirac{T}) [ where ... ]) or Nothing signature") 
    
    where_Ts = Vector{Union{Symbol, Expr}}()
    
    while @capture(msgs_spec, next_msgs_spec_ where { Ts__ })
        append!(where_Ts, Ts)
        msgs_spec = next_msgs_spec
    end
    
    @capture(msgs_spec, (entries__, )) ||
        error("Invalid rule macro call: marginals specification of rule should have a (marginals => (in1::Dirac{T}, in2::Dirac{T}) [ where ... ]) or Nothing signature") 
    
    length(entries) !== 0 || error("Invalid rule macro call: marginals length should be greater than zero")
    
    specs = map(entries) do entry
        @capture(entry, name_::Type_) || 
            error("Invalid rule macro call: marginals specification of rule should have a (marginals => (in1::Dirac{T}, in2::Dirac{T}) [ where ... ]) or Nothing signature") 
        return (name, Type)
    end
    
    all(d -> startswith(string(d[1]), "q_"), specs) || error("Invalid rule macro call: All names in marginals should be prefixed with q_")
    all(d -> length(string(d[1])) > 2, specs) || error("Invalid rule macro call: Empty name in marginals specification")
    
    return where_Ts, specs
end

function __extract_meta_macro_rule(meta::Expr)
    @capture(meta, (meta => Meta_)) ||
        error("Invalid rule macro call: meta specification of rule should have a (meta => Type) signature")
    return Meta
end

function __make_rule(name, fform, on, vconstraint, messages, marginals, meta, fn)
    
    messages_where_Ts, messages_specs = __extract_messages_macro_rule(messages)
    
    messages_names     = messages_specs !== nothing ? map(a -> begin @views Symbol(string(a[1])[3:end]) end, messages_specs) : nothing
    messages_names_val = messages_names !== nothing ? :(Type{ Val{ $(tuple(messages_names...)) } }) : (:(Nothing))
    messages_types     = messages_specs !== nothing ? map(a -> :(Message{ <: $(a[2]) }), messages_specs) : nothing
    messages_types_val = messages_types !== nothing ? :(Tuple{ $(messages_types...) }) : (:(Nothing))
    
    marginals_where_Ts, marginals_specs = __extract_marginals_macro_rule(marginals)
    
    marginals_names     = marginals_specs !== nothing ? map(a -> begin @views Symbol(string(a[1])[3:end]) end, marginals_specs) : nothing
    marginals_names_val = marginals_names !== nothing ? :(Type{ Val{ $(tuple(marginals_names...)) } }) : (:(Nothing))
    marginals_types     = marginals_specs !== nothing ? map(a -> :(Marginal{ <: $(a[2]) }), marginals_specs) : nothing
    marginals_types_val = marginals_types !== nothing ? :(Tuple{ $(marginals_types...) }) : (:(Nothing))
    
    messages_init_block  = messages_names !== nothing ? map(i_name -> :($(Symbol(:m_, i_name[2])) = messages[$(i_name[1])]), enumerate(messages_names)) : [ :nothing ]
    marginals_init_block = marginals_names !== nothing ? map(i_name -> :($(Symbol(:q_, i_name[2])) = marginals[$(i_name[1])]), enumerate(marginals_names)) : [ :nothing ]
    
    result = quote
        function ReactiveMP.$(name)(
            fform::$(__extract_fform_macro_rule(fform)),
            on::$(__extract_on_macro_rule(on)),
            vconstraint::$(__extract_vconstraint_macro_rule(vconstraint)),
            ::$(messages_names_val),
            messages::$(messages_types_val),
            ::$(marginals_names_val),
            marginals::$(marginals_types_val),
            meta::$(__extract_meta_macro_rule(meta))
        ) where { $(messages_where_Ts...), $(marginals_where_Ts...) }
            $(messages_init_block...)
            $(marginals_init_block...)
            $(fn)
        end
    end
    
    return esc(result)
end

macro rule(fform, on, vconstraint, messages, marginals, meta, fn)
    __make_rule(:rule, fform, on, vconstraint, messages, marginals, meta, fn)
end

macro marginalrule(fform, on, messages, marginals, meta, fn)
    __make_rule(:marginalrule, fform, on, :Nothing, messages, marginals, meta, fn)
end