@rule typeof(NOT)(:out, Marginalisation) (
    m_in1::Bernoulli
) = begin
    pin = mean(m_in)  
    return Bernoulli(pin)
end