@rule typeof(AND)(:in1, Marginalisation) (
    m_out::Bernoulli,
    m_in2::Bernoulli
) = begin
    pout, pin2 = mean(m_out), mean(m_in2)
    
    return Bernoulli((1-pout-p2+2*pout*pin2)/(2-2*pout-pin2+2*pout*pin2))
end