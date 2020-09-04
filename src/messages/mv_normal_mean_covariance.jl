import LinearAlgebra: Hermitian

function multiply_messages(m1::Message{N}, m2::Message{N}) where { N <: MvNormalMeanCovariance }
    mean1 = mean(m1)
    mean2 = mean(m2)

    cov1 = cov(m1)
    cov2 = cov(m2)

    cov12inv = inv(cov1 + cov2)

    covr  = Matrix(Hermitian(cov1 * cov12inv * cov2))
    meanr = cov2 * cov12inv * mean1 + cov1 * cov12inv * mean2

    return Message(MvNormalMeanCovariance(meanr, PDMat(covr)))
end