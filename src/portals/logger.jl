export LoggerPortal, apply

struct LoggerPortal <: AbstractPortal end

apply(::LoggerPortal, factornode, tag::Type{ <: Val{ T } },    stream) where { T } = stream |> tap((v) -> Core.println("[Log][$(functionalform(factornode))][$(T)]: $v"))
apply(::LoggerPortal, factornode, tag::Tuple{ Val{ T }, Int }, stream) where { T } = stream |> tap((v) -> Core.println("[Log][$(functionalform(factornode))][$(T):$(tag[2])]: $v"))