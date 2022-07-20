@rule OR(:in1, Marginalisation) (
    m_out::Bernoulli,
    m_in2::Bernoulli
) = begin
    pin2, pout = mean(m_in2), mean(m_out)
    return Bernoulli(pout / (1 - pin2 + 2 * pin2 * pout))
end
