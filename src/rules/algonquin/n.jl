export rule

@rule Algonquin(:n, Marginalisation) (q_out::Any, q_s::NormalDistributionsFamily, q_γ::PointMass) = begin
    
    # fetch outgoing node
    q_n = Rocket.getrecent(getmarginal(ReactiveMP.connectedvar(ReactiveMP.getinterface(__node, :n)), ReactiveMP.IncludeAll()))
    
    # fetch parameters
    mx = mean(q_out)
    ms = mean(q_s)
    mn = mean(q_n)
    γ = mean(q_γ)

    # calculate new parameters
    mn = mn - (log(exp(ms)+exp(mn)) - mx)/(sigmoid(mn-ms))
    wn = γ*sigmoid(mn-ms)^2

    return NormalMeanPrecision(mn, wn)
end
