export ReactiveMPBackend, @model

using GraphPPL

macro model(model_specification)
    return esc(:(@model [] $model_specification))
end

macro model(model_options, model_specification)
    return GraphPPL.generate_model_expression(ReactiveMPBackend(), model_options, model_specification)
end

struct ReactiveMPBackend end

function GraphPPL.write_argument_guard(::ReactiveMPBackend, argument::Symbol)
    return :(@assert !($argument isa ReactiveMP.AbstractVariable) "It is not allowed to pass AbstractVariable objects to a model definition arguments. ConstVariables should be passed as their raw values.")
end

function GraphPPL.write_randomvar_expression(::ReactiveMPBackend, model, varexp, arguments)
    return :($varexp = ReactiveMP.randomvar($model, $(fquote(varexp)), $(arguments...)))
end

function GraphPPL.write_datavar_expression(::ReactiveMPBackend, model, varexpr, type, arguments)
    return :($varexpr = ReactiveMP.datavar($model, $(fquote(varexpr)), ReactiveMP.PointMass{ $type }, $(arguments...)))
end

function GraphPPL.write_constvar_expression(::ReactiveMPBackend, model, varexpr, arguments)
    return :($varexpr = ReactiveMP.constvar($model, $(fquote(varexpr)), $(arguments...)))
end

function GraphPPL.write_as_variable(::ReactiveMPBackend, model, varexpr)
    return :(ReactiveMP.as_variable($model, $varexpr))
end

function GraphPPL.write_make_node_expression(::ReactiveMPBackend, model, fform, variables, options, nodeexpr, varexpr)
    return :($nodeexpr = ReactiveMP.make_node($model, $fform, $varexpr, $(variables...); $(options...)))
end

function GraphPPL.write_autovar_make_node_expression(::ReactiveMPBackend, model, fform, variables, options, nodeexpr, varexpr, autovarid)
    return :(($nodeexpr, $varexpr) = ReactiveMP.make_node($model, $fform, ReactiveMP.AutoVar($(fquote(autovarid))), $(variables...); $(options...)))
end

function GraphPPL.write_node_options(::ReactiveMPBackend, fform, variables, options)
    return map(options) do option

        # Factorisation constraint option
        if @capture(option, q = fconstraint_)
            return write_fconstraint_option(fform, variables, fconstraint)
        elseif @capture(option, meta = fmeta_)
            return write_meta_option(fmeta)
        elseif @capture(option, portal = fportal_)
            return write_portal_option(fportal)
        end

        error("Unknown option '$option' for '$fform' node")
    end
end

# Meta helper functions

function write_meta_option(fmeta)
    return :(meta = $fmeta)
end

# Portal helper functions

function write_portal_option(fportal)
    return :(portal = $fportal)
end

# Factorisation constraint helper functions

function factorisation_replace_var_name(varnames, arg::Expr)
    index = findfirst(==(arg), varnames)
    return index === nothing ? error("Invalid factorisation argument: $arg. $arg should be available within tilde expression") : index
end

function factorisation_replace_var_name(varnames, arg::Symbol)
    index = findfirst(==(arg), varnames)
    return index === nothing ? arg : index
end

function factorisation_name_to_index(form, name)
    return ReactiveMP.interface_get_index(Val{ form }, Val{ ReactiveMP.interface_get_name(Val{ form }, Val{ name }) })
end

function write_fconstraint_option(form, variables, fconstraint)
    if @capture(fconstraint, (*(factors__)) | (q(names__)))
        factors = factors === nothing ? [ fconstraint ] : factors
        indexed = map(factors) do factor
            @capture(factor, q(names__)) || error("Invalid factorisation constraint: $factor")
            return map((n) -> factorisation_name_to_index(form, n), map((n) -> factorisation_replace_var_name(variables, n), names))
        end

        factorisation = sort(map(sort, indexed); by = first)

        allunique(Iterators.flatten(factorisation)) || error("Invalid factorisation constraint: $fconstraint. Arguments are not unique")

        return Expr(:(=), :factorisation, Expr(:tuple, map(f -> Expr(:tuple, f...), factorisation)...))
    elseif @capture(fconstraint, MeanField())
        return :(factorisation = MeanField())
    elseif @capture(fconstraint, FullFactorisation())
        return :(factorisation = FullFactorisation())
    else
        error("Invalid factorisation constraint: $fconstraint")
    end
end