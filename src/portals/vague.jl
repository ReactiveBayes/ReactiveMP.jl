export InitVaguePortal, apply

# TODO, wont' work for nodes with different variate_form on different edges
struct InitVaguePortal{D} <: AbstractPortal 
    dimensionality :: D
end

InitVaguePortal() = InitVaguePortal(nothing)

getdimensionality(portal::InitVaguePortal) = portal.dimensionality

variate_form(portal::InitVaguePortal) = variate_form(getdimensionality(portal), portal)

variate_form(::Nothing,         ::InitVaguePortal) = Univariate
variate_form(::Int,             ::InitVaguePortal) = Multivariate
variate_form(::Tuple{Int, Int}, ::InitVaguePortal) = Matrixvariate

apply(portal::InitVaguePortal, factornode, tag, stream) = apply_vague_portal(variate_form(portal), portal, factornode, tag, stream)

function apply_vague_portal(::Type{ Univariate }, portal::InitVaguePortal, factornode, tag, stream) 
    return stream |> start_with(Message(vague(conjugate_type(functionalform(factornode), tag)), false, true))
end

function apply_vague_portal(::Type{ Multivariate }, portal::InitVaguePortal, factornode, tag, stream) 
    return stream |> start_with(Message(vague(conjugate_type(functionalform(factornode), tag), getdimensionality(portal)), false, true))
end

function apply_vague_portal(::Type{ Matrixvariate }, portal::InitVaguePortal, factornode, tag, stream) 
    return stream |> start_with(Message(vague(conjugate_type(functionalform(factornode), tag), getdimensionality(portal)...), false, true))
end