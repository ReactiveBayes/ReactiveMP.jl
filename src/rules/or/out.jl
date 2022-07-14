@rule typeof(OR)(:out, Marginalisation) (
    m_in1::Bernoulli,
    m_in2::Bernoulli
) = begin
    pin1, pin2 = mean(m_in1), mean(m_in2)
    
    return Bernoulli(pin1 + pin2 - pin1*pin2)
end