export rule

@rule Algonquin(:s, Marginalisation) (q_out::Any, q_n::NormalDistributionsFamily, q_γ::PointMass) = begin
    
    # fetch outgoing node
    q_s = Rocket.getrecent(getmarginal(ReactiveMP.connectedvar(ReactiveMP.getinterface(__node, :s)), ReactiveMP.IncludeAll()))
    
    # fetch parameters
    mx = mean(q_out)
    ms = mean(q_s)
    mn = mean(q_n)
    γ = mean(q_γ)

    # calculate new parameters
    ms = ms - (log(exp(ms)+exp(mn)) - mx)/(sigmoid(ms-mn))
    ws = γ*sigmoid(ms-mn)^2

    return NormalMeanPrecision(ms, ws)
end
