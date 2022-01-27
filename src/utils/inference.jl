import ProgressMeter
export inference

obtain_marginal(variable::AbstractVariable)                      = getmarginal(variable)
obtain_marginal(variables::AbstractArray{ <: AbstractVariable }) = getmarginals(variables)

assign_marginal!(variables::AbstractArray{ <: AbstractVariable }, marginals) = setmarginals!(variables, marginals)
assign_marginal!(variable::AbstractVariable, marginal)                       = setmarginal!(variable, marginal)

assign_message!(variables::AbstractArray{ <: AbstractVariable }, messages) = setmessages!(variables, messages)
assign_message!(variable::AbstractVariable, message)                       = setmessage!(variable, message)

struct KeepEach end
struct KeepLast end

make_actor(::Any, ::KeepEach)                                = nothing
make_actor(::RandomVariable, ::KeepEach)                     = keep(Marginal)
make_actor(::AbstractArray{ <: RandomVariable }, ::KeepEach) = keep(Vector{Marginal})

make_actor(::RandomVariable, ::KeepLast)                     = error("Not implemented")
make_actor(x::AbstractArray{ <: RandomVariable }, ::KeepLast) = buffer(Marginal, length(x))

struct ReturnStructure
    posteriors
    free_energy
end

function inference(; 
    # `model`: specifies a **callback** to create a model, may use whatever global parameters as it wants, required
    model = () -> error("model keyword is required."), 
    # NamedTuple with data, required
    data = nothing,
    # NamedTuple with initial marginals, optional, defaults to empty
    initmarginals = nothing,
    # NamedTuple with initial messages, optional, defaults to empty
    initmessages = nothing,  # optional
    # Reserverd for the future
    constraints = nothing,
    # Return structure info, optional, defaults to return everything at each iteration
    returnvars = nothing, 
    # Number of iterations, defaults to 1, we do not distinguish between VMP or Loopy belief or EP iterations
    iterations = 1,
    # Do we compute FE, optional, defaults to false
    free_energy = false,
    # Show progress module, optional, defaults to false
    showprogress = false,)

    _model, _ = model()
    vardict = ReactiveMP.getvardict(_model)

    # First what we do - we check if `returnvars` is nothing. If so, we replace it with 
    # `KeepEach` for each 
    ireturnvars = if returnvars === nothing 
        Dict(variable => KeepEach() for (variable, value) in vardict)
    else 
        returnvars
    end

    # Second, for each entry we create an actor and we drop `nothing` 
    actors = Dict(variable => make_actor(vardict[variable], value) for (variable, value) in ireturnvars)

    subscriptions = Dict(variable => subscribe!(obtain_marginal(vardict[variable]), actor) for (variable, actor) in actors if actor !== nothing)
    
    fe_actor, fe_subscription = if free_energy
        _fe_actor = ScoreActor()
        _fe_subscription = subscribe!(score(BetheFreeEnergy(), _model), _fe_actor)
        (_fe_actor, _fe_subscription)
    else
        nothing, VoidTeardown()
    end

    if initmarginals !== nothing
        for (variable, initvalue) in initmarginals
            assign_marginal!(vardict[variable], initvalue)
        end
    end

    if initmessages !== nothing
        for (variable, initvalue) in initmessages
            assign_message!(vardict[variable], initvalue)
        end
    end

    if isnothing(data) || isempty(data)
        error("No data provided")
    end
    p = showprogress ? ProgressMeter.Progress(iterations) : nothing
    for _ in 1:iterations
        for (key, value) in data
            update!(vardict[key], value)
        end
        if !isnothing(p)
            ProgressMeter.next!(p)
        end
    end

    for (_, subscription) in subscriptions
        unsubscribe!(subscription)
    end

    unsubscribe!(fe_subscription)
    
    # Todo return proper structure
    return ReturnStructure(actors, fe_actor)
end