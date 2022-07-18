@rule NOT(:in, Marginalisation) (
    m_out::Bernoulli
) = begin
    pout = mean(m_out)
    return Bernoulli(pout)
end