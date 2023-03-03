@rule IMPLY(:in2, Marginalisation) (m_out::Bernoulli, m_in1::Bernoulli) = begin
    pout, pin1 = mean(m_out), mean(m_in1)

    return Bernoulli((pout) / (2 * pout + pin1 - 2 * pout * pin1))
end
