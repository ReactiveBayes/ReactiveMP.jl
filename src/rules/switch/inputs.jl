@rule Switch((:inputs, k), Marginalisation) (m_out::Any, m_switch::Any) where {N} = begin
    @logscale getlogscale(messages[1]) + getlogscale(messages[2]) + log(probvec(messages[2])[k])
    return m_out
end

# type unstable :(
@rule Switch((:inputs, k), Marginalisation) (m_out::Any, q_switch::PointMass) where {N} = begin

    # check whether mean is one-hot
    p = mean(q_switch)
    @assert sum(p) ≈ 1 "The selector variable connected to the switch node is not normalized."
    @assert all(x -> x == 1 || x == 0, p) "The selector variable connected to the switch node is not one-hot encoded."

    # get selected cluster
    kmax = argmax(p)

    if k == kmax
        @logscale 0
        m_out
    else
        _addons = nothing
        Missing()
    end
end
