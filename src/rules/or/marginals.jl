@marginalrule OR(:in1_in2) (
    m_out::Bernoulli,
    m_in1::Bernoulli,
    m_in2::Bernoulli
) = begin
    pin1, pin2, pout = mean(m_in1), mean(m_in2), mean(m_out)
    return Contingency([(1-pin1)*(1-pin2)*(1-pout) (1-pin1)*pin2*pout; pin1*(1-pin2)*pout pin1*pin2*pout])
end
