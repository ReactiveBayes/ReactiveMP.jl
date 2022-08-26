export VariablesCollection, getrandom, getconstant, getdata, getvardict
export hasrandomvar, hasdatavar, hasconstvar

import Base: haskey, getindex, firstindex, lastindex, show

struct VariablesCollection 
    random      :: Vector{RandomVariable}
    constant    :: Vector{ConstVariable}
    data        :: Vector{DataVariable}
    vardict     :: Dict{Symbol, Any}

    VariablesCollection() = new(Vector{RandomVariable}(), Vector{ConstVariable}(), Vector{DataVariable}(), Dict{Symbol, Any}())
end

function Base.show(io::IO, collection::VariablesCollection)
    print(io, "VariablesCollection(random: ", length(getrandom(collection)), ", constant: ", length(getconstant(collection)), ", data: ", length(getdata(collection)), ")")
end

getrandom(collection::VariablesCollection)      = collection.random
getconstant(collection::VariablesCollection)    = collection.constant
getdata(collection::VariablesCollection)        = collection.data
getvardict(collection::VariablesCollection)     = collection.vardict

Base.firstindex(collection::VariablesCollection, symbol::Symbol) = firstindex(collection, getindex(collection, symbol))
Base.lastindex(collection::VariablesCollection, symbol::Symbol)  = lastindex(collection, getindex(collection, symbol))

function Base.getindex(collection::VariablesCollection, symbol::Symbol)
    vardict = getvardict(collection)
    if !haskey(vardict, symbol)
        error("Model has no variable/variables named $(symbol).")
    end
    return getindex(getvardict(collection), symbol)
end

function Base.haskey(collection::VariablesCollection, symbol::Symbol)
    return haskey(getvardict(collection), symbol)
end

hasrandomvar(collection::VariablesCollection, symbol::Symbol) = haskey(collection, symbol) ? israndom(getindex(collection, symbol)) : false
hasdatavar(collection::VariablesCollection, symbol::Symbol)   = haskey(collection, symbol) ? isdata(getindex(collection, symbol)) : false
hasconstvar(collection::VariablesCollection, symbol::Symbol)  = haskey(collection, symbol) ? isconst(getindex(collection, symbol)) : false

Base.push!(::VariablesCollection, ::Nothing) = nothing

function Base.push!(collection::VariablesCollection, randomvar::RandomVariable)
    push!(collection.random, randomvar)
    setindex!(getvardict(collection), name(randomvar), randomvar)
    return randomvar
end

function Base.push!(collection::VariablesCollection, randomvars::AbstractArray{ <: RandomVariable })
    append!(collection.random, randomvars)
    setindex!(getvardict(collection), name(first(randomvars)), randomvars)
    return randomvars
end 

function Base.push!(collection::VariablesCollection, constvar::ConstVariable)
    push!(collection.constant, constvar)
    setindex!(getvardict(collection), name(constvar), constvar)
    return constvar
end

function Base.push!(collection::VariablesCollection, constvars::AbstractArray{ <: ConstVariable })
    append!(collection.constant, constvars)
    setindex!(getvardict(collection), name(first(constvars)), constvars)
    return constvars
end

function Base.push!(collection::VariablesCollection, datavar::DataVariable)
    push!(collection.data, datavar)
    setindex!(getvardict(collection), name(datavar), datavar)
    return datavar
end

function Base.push!(collection::VariablesCollection, datavars::AbstractArray{ <: DataVariable })
    append!(collection.data, datavars)
    setindex!(getvardict(collection), name(first(datavars)), datavars)
    return datavars
end