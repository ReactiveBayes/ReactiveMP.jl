export rule
import LinearAlgebra: dot

@rule typeof(dot)(:out, Marginalisation, symmetrical = [:in1, :in2]) (m_in1::PointMass{ <: AbstractVector }, m_in2::NormalDistributionsFamily) = begin
    NormalMeanVariance(mean(m_in1)'*mean(m_in2), first(mean(m_in1)'*cov(m_in2)*mean(m_in1)))
end

@rule typeof(dot)(:out, Marginalisation) (m_in1::PointMass, m_in2::NormalDistributionsFamily) = begin
    NormalMeanVariance(mean(m_in1)'*mean(m_in2), first(mean(m_in1)'*cov(m_in2)*mean(m_in1)))
end

@rule typeof(dot)(:out, Marginalisation, symmetrical = [:in1, :in2]) (m_in1::NormalDistributionsFamily, m_in2::PointMass{ <: AbstractVector }) = begin
    NormalMeanVariance(mean(m_in2)'*mean(m_in1), first(mean(m_in2)'*cov(m_in1)*mean(m_in2)))
end

@rule typeof(dot)(:out, Marginalisation) (m_in1::NormalDistributionsFamily, m_in2::PointMass) = begin
    NormalMeanVariance(mean(m_in2)'*mean(m_in1), first(mean(m_in2)'*cov(m_in1)*mean(m_in2)))
end