@marginalrule NOT(:in) (m_out::Bernoulli, m_in::Bernoulli) = begin
    pin, pout = mean(m_in), mean(m_out)
    return Bernoulli(pin * (1 - pout) / (pin * (1 - pout) + pout * (1 - pin)))
end
