using MacroTools


function __extract_fform_macro_rule(fformtype)
    if @capture(fformtype, (formtype => typeof(form_)))
        return form
    elseif @capture(fformtype, (formtype => Type{ <: form_ }))
        return form
    elseif @capture(fformtype, (formtype => Type{ form_ }))
        return form
    elseif @capture(fformtype, (formtype => form_))
        return form
    else
        error("Error in macro: functional form of rule should have a (formtype => Type{ <: Distribution } or typeof(fn)) signature")
    end
end

function __extract_fformtype_macro_rule(fformtype)
    if @capture(fformtype, (formtype => typeof(formtype_)))
        return :(typeof($formtype))
    elseif @capture(fformtype, (formtype => Type{ <: formtype_ }))
        return :(Type{ <: $formtype })
    elseif @capture(fformtype, (formtype => Type{ formtype_ }))
        return :(Type{ <: $formtype })
    elseif @capture(fformtype, (formtype => formtype_))
        return :(Type{ <: $formtype })
    else
        error("Error in macro: functional form type should have a (form => Type{ FunctionalFormType } or typeof(fn) for functions) signature")
    end
end

function __extract_sdtype_macro_rule(sdtype)
    if @capture(sdtype, (sdtype => Deterministic))
        return (:Deterministic)
    elseif @capture(sdtype, (sdtype => Stochastic))
        return (:Stochastic)
    else
        error("Error in macro: sdtype specification should have a (sdtype => Deterministic or Stochastic) signature")
    end
end

function __extract_on_macro_rule(on)
    @capture(on, (on => :name_)) ||
        error("Error in macro: on specification should have a (on => :name) signature")
    return :(Type{ Val{ $(QuoteNode(name)) } })
end

function __extract_vconstraint_macro_rule(vconstraint)
    vconstraint !== :Nothing || return vconstraint
    @capture(vconstraint, (vconstraint => Constraint_)) ||
        error("Error in macro: edge specification should have a (vconstraint => Constraint) signature")
    return Constraint
end

function __extract_fn_args_macro_rule(fn_specification; specname, prefix, proxytype)
    if @capture(fn_specification, (name_ => Nothing)) && name === specname
        return :Nothing, :Nothing, [], []
    end
    
    @capture(fn_specification, (name_ => args_specification_)) && name === specname ||
        error("Error in macro: $(specname) specification should have a ($(specname) => (in1::Type1, in2::Type2) [ where ... ]) or Nothing signature") 
    
    where_Ts = []
    
    while @capture(args_specification, next_args_specification_ where { Ts__ })
        append!(where_Ts, Ts)
        args_specification = next_args_specification
    end
    
    @capture(args_specification, (entries__, )) ||
        error("Error in macro: messages specification should have a ($(specname) => (in1::Type1, in2::Type2) [ where ... ]) or Nothing signature") 
    
    length(entries) !== 0 || error("Error in macro: $(specname) length should be greater than zero")
    
    specs = map(entries) do entry
        @capture(entry, name_::Type_) || 
            error("Error in macro: messages specification should have a ($(specname) => (in1::Type1, in2::Type2) [ where ... ]) or Nothing signature") 
        return (name, Type)
    end

    all(d -> startswith(string(d[1]), string(prefix)), specs) || error("Error in macro: All names in $(specname) should be prefixed with $(prefix)")
    all(d -> length(string(d[1])) > 2, specs) || error("Error in macro: Empty name in $(specname) specification")

    args_names     = specs !== nothing ? map(a -> begin @views Symbol(string(a[1])[3:end]) end, specs) : nothing
    args_names_val = args_names !== nothing ? :(Type{ Val{ $(tuple(args_names...)) } }) : (:(Nothing))
    args_types     = specs !== nothing ? map(a -> :($(proxytype){ <: $(a[2]) }), specs) : nothing
    args_types_val = args_types !== nothing ? :(Tuple{ $(args_types...) }) : (:(Nothing))
    init_block     = args_names !== nothing ? map(i_name -> :($(Symbol(prefix, i_name[2])) = getdata($(specname)[$(i_name[1])])), enumerate(args_names)) : [ :nothing ]
    
    return args_names_val, args_types_val, init_block, where_Ts
end

function __extract_meta_macro_rule(meta::Expr)
    @capture(meta, (meta => Meta_)) ||
        error("Invalid rule macro call: meta specification should have a (meta => Type) signature")
    return Meta
end

function __extract_interfaces_macro_rule(interfaces)
    interfacelist = []

    @capture(interfaces, (interfaces => [ args__ ])) ||
        error("Invalid rule macro call: interfaces specification should have a (interfaces => [ ... ]) signature")

    foreach(args) do arg
        if @capture(arg, name_Symbol) 
            push!(interfacelist, (name = name, aliases = []))
        elseif @capture(arg, (name_Symbol, aliases = [ aliases__ ]))
            @assert all(a -> a isa Symbol, aliases)
            push!(interfacelist, (name = name, aliases = aliases))
        else
            error("Invalid macro call: interface specification should have a 'name' or (name, aliases = [ alias1, alias2,... ]) signature")
        end
    end 

    return interfacelist
end