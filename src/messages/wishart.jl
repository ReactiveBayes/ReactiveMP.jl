
function multiply_messages(m1::Message{ <: Wishart }, m2::Message{ <: Wishart })
    # TODO reimplement
    left  = getdata(m1)
    right = getdata(m2)

    d = size(left)[1]

    V  = Matrix(Hermitian(left.S * cholinv(left.S + right.S) * right.S))
    df = left.df + right.df - d - 1.0

    return Message(Wishart(df, V))
end