export InitVaguePortal, apply

struct InitVaguePortal <: AbstractPortal end

apply(::InitVaguePortal, factornode, tag, stream) = stream |> start_with(as_message(vague(conjugate_type(functionalform(factornode), tag))))