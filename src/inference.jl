export inference, KeepEach, KeepLast

import ProgressMeter

obtain_marginal(variable::AbstractVariable)                      = getmarginal(variable)
obtain_marginal(variables::AbstractArray{ <: AbstractVariable }) = getmarginals(variables)

assign_marginal!(variables::AbstractArray{ <: AbstractVariable }, marginals) = setmarginals!(variables, marginals)
assign_marginal!(variable::AbstractVariable, marginal)                       = setmarginal!(variable, marginal)

assign_message!(variables::AbstractArray{ <: AbstractVariable }, messages) = setmessages!(variables, messages)
assign_message!(variable::AbstractVariable, message)                       = setmessage!(variable, message)

struct KeepEach end
struct KeepLast end

make_actor(::RandomVariable, ::KeepEach)                     = keep(Marginal)
make_actor(::AbstractArray{ <: RandomVariable }, ::KeepEach) = keep(Vector{Marginal})

make_actor(::RandomVariable, ::KeepLast)                      = storage(Marginal)
make_actor(x::AbstractArray{ <: RandomVariable }, ::KeepLast) = buffer(Marginal, length(x))

## Inference ensure update

mutable struct MarginalHasBeenUpdated
    updated :: Bool
end

__unset_updated!(updated::MarginalHasBeenUpdated) = updated.updated = false
__set_updated!(updated::MarginalHasBeenUpdated)   = updated.updated = true

# This creates a `tap` operator that will set the `updated` flag to true. 
# Later on we check flags and `unset!` them after the `update!` procedure
ensure_update(model::FactorGraphModel, callback, variable_name::Symbol, updated::MarginalHasBeenUpdated)  = tap((update) -> begin __set_updated!(updated); callback(model, variable_name, update) end)
ensure_update(model::FactorGraphModel, ::Nothing, variable_name::Symbol, updated::MarginalHasBeenUpdated) = tap((_) -> __set_updated!(updated)) # If `callback` is nothing we simply set updated flag

## Extra error handling

__inference_process_error(error) = rethrow(error)

function __inference_process_error(err::StackOverflowError)
    error("""
        Stack overflow error occurred during the inference procedure. 
        The dataset size might be causing this error. 
        To circumvent this behavior, try using `limit_stack_depth` option when creating a model.
    """)
end
##

struct InferenceResult{P, F}
    posteriors  :: P
    free_energy :: F
end

Base.iterate(results::InferenceResult)      = iterate((getfield(results, :posteriors), getfield(results, :free_energy)))
Base.iterate(results::InferenceResult, any) = iterate((getfield(results, :posteriors), getfield(results, :free_energy)), any)

function Base.show(io::IO, result::InferenceResult)
    print(io, "Inference results:\n")
    if !isnothing(getfield(result, :free_energy))
        print(io, "-----------------------------------------\n")
        print(io, "Free Energy: ")
        print(IOContext(io, :compact => true, :limit => true, :displaysize => (1, 80)), result.free_energy)
        print(io, "\n")
    end

    maxwidth = 80
    maxlen   = maximum(p -> length(string(first(p))), pairs(result.posteriors))
    print(io, "-----------------------------------------\n")
    for (key, value) in pairs(result.posteriors)
        print(io, "$(rpad(key, maxlen)) = ")
        svalue = string(value)
        slen   = length(svalue)
        print(IOContext(io, :compact => true, :limit => true), view(svalue, 1:min(maxwidth, length(svalue))))
        if slen > maxwidth
            print(io, "...")
        end
        print(io, "\n")
    end
end

function Base.getproperty(result::InferenceResult, property::Symbol)
    if property === :free_energy && getfield(result, :free_energy) === nothing 
        error("""
            Bethe Free Energy has not been computed. 
            Use `free_energy = true` keyword argument for the `inference` function to compute Bethe Free Energy values.
        """)
    else
        return getfield(result, property)
    end
    return getfield(result, property)
end



__inference_invoke_callback(callback, args...)  = callback(args...)
__inference_invoke_callback(::Nothing, args...) = begin end

inference_invoke_callback(callbacks, name, args...) = __inference_invoke_callback(inference_get_callback(callbacks, name), args...)
inference_invoke_callback(::Nothing, name, args...) = begin end

inference_get_callback(callbacks, name) = get(() -> nothing, callbacks, name)
inference_get_callback(::Nothing, name) = nothing

"""
    inference(
        # `model`: specifies a model generator, with the help of the `Model` function
        model::ModelGenerator; 
        # NamedTuple or Dict with data, required
        data,
        # NamedTuple with initial marginals, optional, defaults to empty
        initmarginals = nothing,
        # NamedTuple with initial messages, optional, defaults to empty
        initmessages = nothing,  # optional
        # Constraints specification object
        constraints = nothing,
        # Meta specification object
        meta  = nothing,
        # Model creation options
        options = (;),
        # Return structure info, optional, defaults to return everything at each iteration
        returnvars = nothing, 
        # Number of iterations, defaults to 1, we do not distinguish between VMP or Loopy belief or EP iterations
        iterations = 1,
        # Do we compute FE, optional, defaults to false
        # Can be passed a floating point type, e.g. `Float64`, for better efficiency, but disables automatic differentiation packages, such as ForwardDiff.jl
        free_energy = false,
        # Show progress module, optional, defaults to false
        showprogress = false,
        # Inference cycle callbacks
        callbacks = nothing,
    )

This function provides generic (but somewhat limited) way to run inference in ReactiveMP.jl. 

## Callbacks

- `:on_marginal_update`:    args: (model::FactorGraphModel, name::Symbol, update)
- `:before_model_creation`: args: ()
- `:after_model_creation`:  args: (model::FactorGraphModel)
- `:before_inference`:      args: (model::FactorGraphModel)
- `:before_iteration`:      args: (model::FactorGraphModel, iteration::Int)
- `:before_data_update`:    args: (model::FactorGraphModel, data)
- `:after_data_update`:     args: (model::FactorGraphModel, data)
- `:after_iteration`:       args: (model::FactorGraphModel, iteration::Int)
- `:after_inference`:       args: (model::FactorGraphModel)
"""
function inference(;
    # `model`: specifies a model generator, with the help of the `Model` function
    model::ModelGenerator,
    # NamedTuple or Dict with data, required
    data,
    # NamedTuple or Dict with initial marginals, optional, defaults to empty
    initmarginals = nothing,
    # NamedTuple or Dict with initial messages, optional, defaults to empty
    initmessages = nothing,  # optional
    # Constraints specification object
    constraints = nothing,
    # Meta specification object
    meta  = nothing,
    # Model creation options
    options = (;),
    # Return structure info, optional, defaults to return everything at each iteration
    returnvars = nothing, 
    # Number of iterations, defaults to 1, we do not distinguish between VMP or Loopy belief or EP iterations
    iterations = 1,
    # Do we compute FE, optional, defaults to false 
    # Can be passed a floating point type, e.g. `Float64`, for better efficiency, but disables automatic differentiation packages, such as ForwardDiff.jl
    free_energy = false,
    # Show progress module, optional, defaults to false
    showprogress = false,
    # Inference cycle callbacks
    callbacks = nothing,
)

    inference_invoke_callback(callbacks, :before_model_creation)
    fmodel, _ = create_model(model, constraints, meta, options)
    inference_invoke_callback(callbacks, :after_model_creation, fmodel)
    vardict = ReactiveMP.getvardict(fmodel)

    # First what we do - we check if `returnvars` is nothing. If so, we replace it with 
    # `KeepEach` for each random and not-proxied variable in a model
    if returnvars === nothing 
        returnvars = Dict(variable => KeepEach() for (variable, value) in pairs(vardict) if (israndom(value) && !isproxy(value)))
    end

    # Second, for each random variable entry we create an actor
    actors  = Dict(variable => make_actor(vardict[variable], value) for (variable, value) in pairs(returnvars))

    # At third, for each random variable entry we create a boolean flag to track their updates
    updates = Dict(variable => MarginalHasBeenUpdated(false) for (variable, _) in pairs(returnvars))

    try 
        on_marginal_update = inference_get_callback(callbacks, :on_marginal_update)
        subscriptions      = Dict(variable => subscribe!(obtain_marginal(vardict[variable]) |> ensure_update(fmodel, on_marginal_update, variable, updates[variable]), actor) for (variable, actor) in pairs(actors))
        
        fe_actor        = nothing
        fe_subscription = VoidTeardown()
        
        if free_energy
            fe_actor        = ScoreActor()
            fe_subscription = subscribe!(score(BetheFreeEnergy(), fmodel), fe_actor)
        end

        if !isnothing(initmarginals)
            for (variable, initvalue) in pairs(initmarginals)
                assign_marginal!(vardict[variable], initvalue)
            end
        end

        if !isnothing(initmessages)
            for (variable, initvalue) in pairs(initmessages)
                assign_message!(vardict[variable], initvalue)
            end
        end

        if isnothing(data) || isempty(data)
            error("Data is empty. Make sure you used `data` keyword argument with correct value.")
        else 
            foreach(filter(pair -> first(pair) isa DataVariable, pairs(vardict))) do pair
                haskey(data, name(pair)) || error("Data entry $(name(p)) is missing in `data` dictionary.")
            end
        end

        p = showprogress ? ProgressMeter.Progress(iterations) : nothing

        inference_invoke_callback(callbacks, :before_inference, fmodel)

        for iteration in 1:iterations
            inference_invoke_callback(callbacks, :before_iteration, fmodel, iteration)
            inference_invoke_callback(callbacks, :before_data_update, fmodel, data)
            for (key, value) in pairs(data)
                update!(vardict[key], value)
            end
            inference_invoke_callback(callbacks, :after_data_update, fmodel, data)
            not_updated = filter((pair) -> !last(pair).updated, updates)
            if length(not_updated) !== 0
                names = join(keys(not_updated), ", ")
                error("""
                    Variables [ $(names) ] have not been updated after a single inference iteration. 
                    Therefore, make sure to initialize all required marginals and messages. See `initmarginals` and `initmessages` keyword arguments for the `inference` function. 
                """)
            end
            for (_, update_flag) in pairs(updates)
                __unset_updated!(update_flag)
            end
            if !isnothing(p)
                ProgressMeter.next!(p)
            end
            inference_invoke_callback(callbacks, :after_iteration, fmodel, iteration)
        end

        for (_, subscription) in pairs(subscriptions)
            unsubscribe!(subscription)
        end

        unsubscribe!(fe_subscription)

        posterior_values = Dict(variable => getvalues(actor) for (variable, actor) in pairs(actors))
        fe_values        = fe_actor !== nothing ? getvalues(fe_actor) : nothing

        inference_invoke_callback(callbacks, :after_inference, fmodel)

        return InferenceResult(posterior_values, fe_values)
    catch error
        __inference_process_error(error)
    end    
end