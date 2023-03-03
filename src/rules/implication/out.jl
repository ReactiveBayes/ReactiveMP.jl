@rule IMPLY(:out, Marginalisation) (m_in1::Bernoulli, m_in2::Bernoulli) = begin
    pin1, pin2 = mean(m_in1), mean(m_in2)

    return Bernoulli(1 - pin1 + pin1 * pin2)
end
