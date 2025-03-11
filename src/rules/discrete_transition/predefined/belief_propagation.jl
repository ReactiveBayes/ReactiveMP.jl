using Tullio

# --------------- Rules for 2 interfaces (PointMass q_a) ---------------
@rule DiscreteTransition(:out, Marginalisation) (m_in::DiscreteNonParametric, q_a::PointMass, meta::Any) = begin
    eloga = mean(q_a)
    @tullio out[i] := eloga[i, a] * probvec(m_in)[a]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:in, Marginalisation) (m_out::DiscreteNonParametric, q_a::PointMass, meta::Any) = begin
    eloga = mean(q_a)
    @tullio out[i] := eloga[a, i] * probvec(m_out)[a]
    return Categorical(normalize!(out, 1))
end

# --------------- Rules for 2 interfaces (DirichletCollection q_a) ---------------
@rule DiscreteTransition(:out, Marginalisation) (m_in::DiscreteNonParametric, q_a::DirichletCollection, meta::Any) = begin
    eloga = exp.(mean(Base.Broadcast.BroadcastFunction(clamplog), q_a))
    @tullio out[i] := eloga[i, a] * probvec(m_in)[a]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:in, Marginalisation) (m_out::DiscreteNonParametric, q_a::DirichletCollection, meta::Any) = begin
    eloga = exp.(mean(Base.Broadcast.BroadcastFunction(clamplog), q_a))
    @tullio out[i] := eloga[a, i] * probvec(m_out)[a]
    return Categorical(normalize!(out, 1))
end

# --------------- Rules for 3 interfaces (PointMass q_a) ---------------
@rule DiscreteTransition(:out, Marginalisation) (m_in::DiscreteNonParametric, m_T1::DiscreteNonParametric, q_a::PointMass, meta::Any) = begin
    eloga = mean(q_a)
    @tullio out[i] := eloga[i, a, b] * probvec(m_in)[a] * probvec(m_T1)[b]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:in, Marginalisation) (m_out::DiscreteNonParametric, m_T1::DiscreteNonParametric, q_a::PointMass, meta::Any) = begin
    eloga = mean(q_a)
    @tullio out[i] := eloga[a, i, b] * probvec(m_out)[a] * probvec(m_T1)[b]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:T1, Marginalisation) (m_out::DiscreteNonParametric, m_in::DiscreteNonParametric, q_a::PointMass, meta::Any) = begin
    eloga = mean(q_a)
    @tullio out[i] := eloga[a, b, i] * probvec(m_out)[a] * probvec(m_in)[b]
    return Categorical(normalize!(out, 1))
end

# --------------- Rules for 3 interfaces (DirichletCollection q_a) ---------------
@rule DiscreteTransition(:out, Marginalisation) (m_in::DiscreteNonParametric, m_T1::DiscreteNonParametric, q_a::DirichletCollection, meta::Any) = begin
    eloga = exp.(mean(Base.Broadcast.BroadcastFunction(clamplog), q_a))
    @tullio out[i] := eloga[i, a, b] * probvec(m_in)[a] * probvec(m_T1)[b]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:in, Marginalisation) (m_out::DiscreteNonParametric, m_T1::DiscreteNonParametric, q_a::DirichletCollection, meta::Any) = begin
    eloga = exp.(mean(Base.Broadcast.BroadcastFunction(clamplog), q_a))
    @tullio out[i] := eloga[a, i, b] * probvec(m_out)[a] * probvec(m_T1)[b]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:T1, Marginalisation) (m_out::DiscreteNonParametric, m_in::DiscreteNonParametric, q_a::DirichletCollection, meta::Any) = begin
    eloga = exp.(mean(Base.Broadcast.BroadcastFunction(clamplog), q_a))
    @tullio out[i] := eloga[a, b, i] * probvec(m_out)[a] * probvec(m_in)[b]
    return Categorical(normalize!(out, 1))
end

# --------------- Rules for 4 interfaces (PointMass q_a) ---------------
@rule DiscreteTransition(:out, Marginalisation) (m_in::DiscreteNonParametric, m_T1::DiscreteNonParametric, m_T2::DiscreteNonParametric, q_a::PointMass, meta::Any) = begin
    eloga = mean(q_a)
    @tullio out[i] := eloga[i, a, b, c] * probvec(m_in)[a] * probvec(m_T1)[b] * probvec(m_T2)[c]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:in, Marginalisation) (m_out::DiscreteNonParametric, m_T1::DiscreteNonParametric, m_T2::DiscreteNonParametric, q_a::PointMass, meta::Any) = begin
    eloga = mean(q_a)
    @tullio out[i] := eloga[a, i, b, c] * probvec(m_out)[a] * probvec(m_T1)[b] * probvec(m_T2)[c]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:T1, Marginalisation) (m_out::DiscreteNonParametric, m_in::DiscreteNonParametric, m_T2::DiscreteNonParametric, q_a::PointMass, meta::Any) = begin
    eloga = mean(q_a)
    @tullio out[i] := eloga[a, b, i, c] * probvec(m_out)[a] * probvec(m_in)[b] * probvec(m_T2)[c]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:T2, Marginalisation) (m_out::DiscreteNonParametric, m_in::DiscreteNonParametric, m_T1::DiscreteNonParametric, q_a::PointMass, meta::Any) = begin
    eloga = mean(q_a)
    @tullio out[i] := eloga[a, b, c, i] * probvec(m_out)[a] * probvec(m_in)[b] * probvec(m_T1)[c]
    return Categorical(normalize!(out, 1))
end

# --------------- Rules for 4 interfaces (DirichletCollection q_a) ---------------
@rule DiscreteTransition(:out, Marginalisation) (m_in::DiscreteNonParametric, m_T1::DiscreteNonParametric, m_T2::DiscreteNonParametric, q_a::DirichletCollection, meta::Any) = begin
    eloga = exp.(mean(Base.Broadcast.BroadcastFunction(clamplog), q_a))
    @tullio out[i] := eloga[i, a, b, c] * probvec(m_in)[a] * probvec(m_T1)[b] * probvec(m_T2)[c]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:in, Marginalisation) (m_out::DiscreteNonParametric, m_T1::DiscreteNonParametric, m_T2::DiscreteNonParametric, q_a::DirichletCollection, meta::Any) = begin
    eloga = exp.(mean(Base.Broadcast.BroadcastFunction(clamplog), q_a))
    @tullio out[i] := eloga[a, i, b, c] * probvec(m_out)[a] * probvec(m_T1)[b] * probvec(m_T2)[c]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:T1, Marginalisation) (m_out::DiscreteNonParametric, m_in::DiscreteNonParametric, m_T2::DiscreteNonParametric, q_a::DirichletCollection, meta::Any) = begin
    eloga = exp.(mean(Base.Broadcast.BroadcastFunction(clamplog), q_a))
    @tullio out[i] := eloga[a, b, i, c] * probvec(m_out)[a] * probvec(m_in)[b] * probvec(m_T2)[c]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:T2, Marginalisation) (m_out::DiscreteNonParametric, m_in::DiscreteNonParametric, m_T1::DiscreteNonParametric, q_a::DirichletCollection, meta::Any) = begin
    eloga = exp.(mean(Base.Broadcast.BroadcastFunction(clamplog), q_a))
    @tullio out[i] := eloga[a, b, c, i] * probvec(m_out)[a] * probvec(m_in)[b] * probvec(m_T1)[c]
    return Categorical(normalize!(out, 1))
end

# --------------- Rules for 5 interfaces (PointMass q_a) ---------------
@rule DiscreteTransition(:out, Marginalisation) (
    m_in::DiscreteNonParametric, m_T1::DiscreteNonParametric, m_T2::DiscreteNonParametric, m_T3::DiscreteNonParametric, q_a::PointMass, meta::Any
) = begin
    eloga = mean(q_a)
    @tullio out[i] := eloga[i, a, b, c, d] * probvec(m_in)[a] * probvec(m_T1)[b] * probvec(m_T2)[c] * probvec(m_T3)[d]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:in, Marginalisation) (
    m_out::DiscreteNonParametric, m_T1::DiscreteNonParametric, m_T2::DiscreteNonParametric, m_T3::DiscreteNonParametric, q_a::PointMass, meta::Any
) = begin
    eloga = mean(q_a)
    @tullio out[i] := eloga[a, i, b, c, d] * probvec(m_out)[a] * probvec(m_T1)[b] * probvec(m_T2)[c] * probvec(m_T3)[d]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:T1, Marginalisation) (
    m_out::DiscreteNonParametric, m_in::DiscreteNonParametric, m_T2::DiscreteNonParametric, m_T3::DiscreteNonParametric, q_a::PointMass, meta::Any
) = begin
    eloga = mean(q_a)
    @tullio out[i] := eloga[a, b, i, c, d] * probvec(m_out)[a] * probvec(m_in)[b] * probvec(m_T2)[c] * probvec(m_T3)[d]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:T2, Marginalisation) (
    m_out::DiscreteNonParametric, m_in::DiscreteNonParametric, m_T1::DiscreteNonParametric, m_T2::DiscreteNonParametric, q_a::PointMass, meta::Any
) = begin
    eloga = mean(q_a)
    @tullio out[i] := eloga[a, b, c, i, d] * probvec(m_out)[a] * probvec(m_in)[b] * probvec(m_T1)[c] * probvec(m_T2)[d]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:T3, Marginalisation) (
    m_out::DiscreteNonParametric, m_in::DiscreteNonParametric, m_T1::DiscreteNonParametric, m_T2::DiscreteNonParametric, q_a::PointMass, meta::Any
) = begin
    eloga = mean(q_a)
    @tullio out[i] := eloga[a, b, c, d, i] * probvec(m_out)[a] * probvec(m_in)[b] * probvec(m_T1)[c] * probvec(m_T2)[d]
    return Categorical(normalize!(out, 1))
end

# --------------- Rules for 5 interfaces (DirichletCollection q_a) ---------------
@rule DiscreteTransition(:out, Marginalisation) (
    m_in::DiscreteNonParametric, m_T1::DiscreteNonParametric, m_T2::DiscreteNonParametric, m_T3::DiscreteNonParametric, q_a::DirichletCollection, meta::Any
) = begin
    eloga = exp.(mean(Base.Broadcast.BroadcastFunction(clamplog), q_a))
    @tullio out[i] := eloga[i, a, b, c, d] * probvec(m_in)[a] * probvec(m_T1)[b] * probvec(m_T2)[c] * probvec(m_T3)[d]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:in, Marginalisation) (
    m_out::DiscreteNonParametric, m_T1::DiscreteNonParametric, m_T2::DiscreteNonParametric, m_T3::DiscreteNonParametric, q_a::DirichletCollection, meta::Any
) = begin
    eloga = exp.(mean(Base.Broadcast.BroadcastFunction(clamplog), q_a))
    @tullio out[i] := eloga[a, i, b, c, d] * probvec(m_out)[a] * probvec(m_T1)[b] * probvec(m_T2)[c] * probvec(m_T3)[d]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:T1, Marginalisation) (
    m_out::DiscreteNonParametric, m_in::DiscreteNonParametric, m_T2::DiscreteNonParametric, m_T3::DiscreteNonParametric, q_a::DirichletCollection, meta::Any
) = begin
    eloga = exp.(mean(Base.Broadcast.BroadcastFunction(clamplog), q_a))
    @tullio out[i] := eloga[a, b, i, c, d] * probvec(m_out)[a] * probvec(m_in)[b] * probvec(m_T2)[c] * probvec(m_T3)[d]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:T2, Marginalisation) (
    m_out::DiscreteNonParametric, m_in::DiscreteNonParametric, m_T1::DiscreteNonParametric, m_T2::DiscreteNonParametric, q_a::DirichletCollection, meta::Any
) = begin
    eloga = exp.(mean(Base.Broadcast.BroadcastFunction(clamplog), q_a))
    @tullio out[i] := eloga[a, b, c, i, d] * probvec(m_out)[a] * probvec(m_in)[b] * probvec(m_T1)[c] * probvec(m_T2)[d]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:T3, Marginalisation) (
    m_out::DiscreteNonParametric, m_in::DiscreteNonParametric, m_T1::DiscreteNonParametric, m_T2::DiscreteNonParametric, q_a::DirichletCollection, meta::Any
) = begin
    eloga = exp.(mean(Base.Broadcast.BroadcastFunction(clamplog), q_a))
    @tullio out[i] := eloga[a, b, c, d, i] * probvec(m_out)[a] * probvec(m_in)[b] * probvec(m_T1)[c] * probvec(m_T2)[d]
    return Categorical(normalize!(out, 1))
end