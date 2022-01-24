obtain_marginal!(variable::AbstractVariable, marginal) = getmarginal!(variables)
obtain_marginal!(variable::AbstractArray{ <: AbstractVariable }, marginal) = getmarginals!(variables)


assign_marginal!(variables::AbstractArray{ <: AbstractVariable }, marginals) = setmarginals!(variables, marginals)
assign_marginal!(variable::AbstractVariable, marginal) = setmarginal!(variable, marginal)


function inference(; kwargs...)

    model, _ = kwargs[:model]()

   # maybe try catch and unsubsribe?
    
   # Check how to map over named tuple values with preserving the same structure
    actors = map(kwargs[:returnvars]) do (variable, strategy) # eg. (x, KeepEach)
        return make_actor(variable, strategy) # returns eg. buffer(Marginal, n) or KeepActor(), dispatch on strategy
    end
    
    # `actors` should be a named tuple, eg. (x = keep(Marginal), z = buffer(Marginal, 10))
    subscriptions = map(actors) do (variable, actor)
        return subscribe!(obtain_marginal!(model.variables[variable]), actor) # obtain_marginal! calls either `getmarginal` or `getmarginals`
    end

    # `initmarginals` is a named tuple, eg. (x = vague(Gamma), z = [ ... ])
    foreach(initmarginals) do (variable, initvalue)
        assign_marginal!(model.variables[variable], initvalue) # __setmarginal calls either `setmarginal!` or `setmarginals!`
    end

    # exactly the same for `initmessages` here

    if kwargs[:free_energy]
        fe_subscription = ...
    else
        fe_subscription = voidTeardown # empty subscription from Rocket.jl, does nothing on `unsubscribe!`
    end

    # showprogress stuff here
    for iteration in kwargs[:iteration]
         foreach(kwargs[:data]) do (variable, data)
             update!(model.variables[variable], data)
         end
    end

    unsubscribe!(subscriptions)
    unsubscribe!(fe_subscription)
    
    return ReturnStructure(actors..., free_energy...) # ??
end