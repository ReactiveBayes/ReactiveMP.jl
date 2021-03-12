export InitVaguePortal, apply

struct InitVaguePortal <: AbstractPortal end

apply(::InitVaguePortal, factornode, tag, stream) = stream |> start_with(Message(vague(conjugate_type(functionalform(factornode), tag))), false, true)