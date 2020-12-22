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

function __apply_proxy_type(type::Symbol, proxytype)
    return :($(proxytype){ <: $(type) })
end

function __apply_proxy_type(type::Expr, proxytype)
    if @capture(type, NTuple{N_, T_})
        return :(NTuple{ $N, <: $(__apply_proxy_type(T, proxytype)) })
    elseif @capture(type, Tuple{ T__ })
        return :(Tuple{ $(map(t -> :( <: $(__apply_proxy_type(t, proxytype))), T)...) })
    else 
        return :($(proxytype){ <: $(type) })
    end
end

function __extract_fformtype(fform)
    if @capture(fform, typeof(f_))
        return fform
    elseif @capture(fform, Type{ T_ })
        return :(Type{ <: $T })
    elseif @capture(fform, Type{ <: T_ })
        return :(Type{ <: $T })
    elseif @capture(fform, T_)
        return :(Type{ <: $T })
    else
        error("Error in macro. fform specification is incorrect")
    end
end

function __extract_on_args_macro_rule(on)
    if @capture(on, :name_)
        return :(Type{ Val{ $(QuoteNode(name)) } }), nothing
    elseif @capture(on, (:name_, index_))
        return :(Tuple{ Val{$(QuoteNode(name))}, Int}), index
    else
        error("Error in macro. on specification is incorrect")
    end
end

function __extract_fn_args_macro_rule(inputs; specname, prefix, proxytype)
    finputs = filter((i) -> startswith(string(first(i)), string(prefix)), inputs)

    names  = map(first, finputs)
    types  = map((i) -> __apply_proxy_type(last(i), proxytype), finputs)

    @assert all((n) -> length(string(n)) > 2, names)  || error("Empty $(specname) name found in arguments")

    init_block = map(enumerate(names)) do (index, iname)
        return :($(iname) = getdata($(specname)[$(index)]))
    end

    out_names = length(names) === 0 ? :Nothing : :(Type{ Val{ $(tuple(map(n -> Symbol(string(n)[(length(string(prefix)) + 1):end]), names)...)) } })
    out_types = length(types) === 0 ? :Nothing : :(Tuple{ $(types...) })

    return out_names, out_types, init_block
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