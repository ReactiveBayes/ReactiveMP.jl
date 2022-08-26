export VariablesCollection, getrandom, getconstant, getdata, getvardict
export hasrandomvar, hasdatavar, hasconstvar

import Base: haskey

struct VariablesCollection 
    random      :: Vector{RandomVariable}
    constant    :: Vector{ConstVariable}
    data        :: Vector{DataVariable}
    vardict     :: Dict{Symbol, Any}
end

getrandom(collection::VariablesCollection)      = collection.random
getconstant(collection::VariablesCollection)    = collection.constant
getdata(collection::VariablesCollection)        = collection.data
getvardict(collection::VariablesCollection)     = collection.vardict

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