@marginalrule NOT(:in1) (
    m_out::Bernoulli,
    m_in1::Bernoulli
) = begin
    pin, pout = mean(m_in1), mean(m_out)
    return Bernoulli(pin * (1 - pout) / (pin * (1 - pout) + pout * (1 - pin)))
end
