export rule, marginalrule
export @rule, @marginalrule

using MacroTools
using .MacroHelpers

import Base: showerror

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

struct RuleMethodError
    fform
    on
    vconstraint
    mnames
    messages
    qnames
    marginals
    meta
    node
end

rule(fform, on, vconstraint, mnames, messages, qnames, marginals, meta, __node) = throw(RuleMethodError(fform, on, vconstraint, mnames, messages, qnames, marginals, meta, __node))

function Base.showerror(io::IO, error::RuleMethodError)

    rule_method_error_extract_fform(f::Function) = string("typeof(", f, ")")
    rule_method_error_extract_fform(f)           = string(f)

    rule_method_error_extract_on(::Type{ Val{ T } }) where T = T

    rule_method_error_extract_vconstraint(something) = typeof(something)

    rule_method_error_extract_names(::Type{ Val{ T } }) where T = T
    rule_method_error_extract_names(::Nothing)                  = ()
    
    rule_method_error_extract_types(t::Tuple)   = map(e -> typeof(getdata(e)), t)
    rule_method_error_extract_types(t::Nothing) = ()

    rule_method_error_extract_meta(something) = string("meta::", typeof(something))
    rule_method_error_extract_meta(::Nothing) = ""

    print(io, "RuleMethodError: no method matching rule for the given arguments")

    try 
        spec_fform       = rule_method_error_extract_fform(error.fform)
        spec_on          = rule_method_error_extract_on(error.on)
        spec_vconstraint = rule_method_error_extract_vconstraint(error.vconstraint)

        m_names   = rule_method_error_extract_names(error.mnames)
        m_indices = map(n -> interfaceindex(error.node, n), m_names)
        q_names   = rule_method_error_extract_names(error.qnames)
        q_indices = map(n -> interfaceindex(error.node, n), q_names)

        spec_m_names = map(e -> string("m_", e), m_names)
        spec_m_types = rule_method_error_extract_types(error.messages)
        spec_q_names = map(e -> string("q_", e), q_names)
        spec_q_types = rule_method_error_extract_types(error.marginals)

        spec_m = map(m -> string(m[1], "::", m[2]), zip(spec_m_names, spec_m_types))
        spec_q = map(q -> string(q[1], "::", q[2]), zip(spec_q_names, spec_q_types))

        spec = Vector(undef, length(interfaces(error.node)))

        fill!(spec, nothing)

        
        for (i, j) in enumerate(m_indices)
            spec[j] = spec_m[i]
        end

        for (i, j) in enumerate(q_indices)
            spec[j] = spec_q[i]
        end

        filter!(!isnothing, spec)

        arguments_spec = join(spec, ",")
        meta_spec      = rule_method_error_extract_meta(error.meta)

        possible_fix_definition = """
        @rule $(spec_fform)(:$spec_on, $spec_vconstraint) ($arguments_spec, $meta_spec) = begin 
            return ...
        end
        """

        println(io, "\n\nPossible fix, define:\n")
        println(io, possible_fix_definition)
    catch uerror
        println(io, "Unexpected error occured during Base.showerror(::RuleMethodError)")
        println(io, uerror)
    end
end



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
        error("Error in macro. Lambda body arguments speicifcation is incorrect")

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
    m_values_arg = isempty(m_names) ? :nothing : :($(map(m_value -> :(ReactiveMP.Message($m_value, false, false)), m_values)...), )
    q_names_arg  = isempty(q_names) ? :nothing : :(Val{ $(q_names) })
    q_values_arg = isempty(q_names) ? :nothing : :($(map(q_value -> :(ReactiveMP.Marginal($q_value, false, false)), q_values)...), )

    on_arg = MacroHelpers.bottom_type(on_type)

    output = quote
        ReactiveMP.rule($fbottomtype, $on_arg, $(vconstraint)(), $m_names_arg, $m_values_arg, $q_names_arg, $q_values_arg, $meta, nothing)
    end

    return esc(output)
end

"""
    Documentation placeholder
"""
macro test_rules(options, on, test_sequence)
    
    @capture(options, [ option_entries__ ]) || error("Invalid options specification. Options should be in the form on an array.")
    
    with_float_conversions = false
    
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
        else 
            error("Unknown option $(key)")
        end
    end
    
    @capture(test_sequence, [ test_sequence_entries__ ]) || error("Invalid test sequence specification. Test sequence should be in the form of an array.")
    
    block      = Expr(:block)
    block.args = map(test_sequence_entries) do test_entry 
        @capture(test_entry, (input = input_, output = output_)) || error("Invalid test entry specification: $(test_entry). Test entry should be in the form of a named tuple (input = ..., output = ...).")
        
        test_rule      = Expr(:block)
        test_output_s  = gensym()
        test_rule.args = [
            quote 
                begin
                    local $test_output_s = ReactiveMP.@call_rule($on, $input)
                    @test isapprox( $test_output_s, $output; atol = 1e-12) 
                    @test ReactiveMP.is_typeof_equal($test_output_s, $output)
                end
            end
        ]
        
        if with_float_conversions
            @capture(input, (input_entries__, )) || error("Invalid input entries. Input entries should be in the form of a named tuple. ")
            
            # We filter out indices only for inputs that start with 'm_' or 'q_'
            inputs = map(first, filter(collect(enumerate(input_entries))) do i
                @capture(i[2], (key_ = value_))
                if key !== nothing
                    skey = string(key)
                    return startswith(skey, "m_") || startswith(skey, "q_")
                end
                return false
            end)

            function powerset(x::Vector{T}) where T
                result = Vector{T}[[]]
                for elem in x, j in eachindex(result)
                    push!(result, [result[j] ; elem])
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
                        @test isapprox($output_s, $m_f32_output; atol = 1e-6)     
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
                output_s = gensym()
                push!(test_rule.args, quote
                    begin 
                        local $output_s = ReactiveMP.@call_rule($on, $(m_bigf_input[1]))
                        @test isapprox($output_s, $m_bigf_output; atol = 1e-12)     
                        @test ReactiveMP.is_typeof_equal($output_s, $m_bigf_output)
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