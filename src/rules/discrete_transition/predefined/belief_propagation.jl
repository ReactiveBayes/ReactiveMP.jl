using Tullio

# # --------------- Rules for 2 interfaces (PointMass q_a) ---------------
@rule DiscreteTransition(:out, Marginalisation) (m_in::DiscreteNonParametric, q_a::PointMass{<:AbstractArray{T, 2}}, meta::Any) where {T} = begin
    N = eltype(probvec(m_in))
    eloga = clamp.(mean(q_a), tiny(N), one(N))
    out = eloga * probvec(m_in)

    return Categorical(normalize!(out, 1); check_args = false)
end

@rule DiscreteTransition(:in, Marginalisation) (m_out::DiscreteNonParametric, q_a::PointMass{<:AbstractArray{T, 2}}, meta::Any) where {T} = begin
    N = eltype(probvec(m_out))
    eloga = clamp.(mean(q_a), tiny(N), one(N))
    out = eloga' * probvec(m_out)
    return Categorical(normalize!(out, 1); check_args = false)
end

# --------------- Rules for 2 interfaces (DirichletCollection q_a) ---------------
@rule DiscreteTransition(:out, Marginalisation) (m_in::DiscreteNonParametric, q_a::DirichletCollection, meta::Any) = begin
    N = eltype(probvec(m_in))
    eloga = softmax!(mean(Base.Broadcast.BroadcastFunction(clamplog), q_a))
    out = eloga * probvec(m_in)
    return Categorical(normalize!(out, 1); check_args = false)
end

@rule DiscreteTransition(:in, Marginalisation) (m_out::DiscreteNonParametric, q_a::DirichletCollection, meta::Any) = begin
    eloga = softmax!(mean(Base.Broadcast.BroadcastFunction(clamplog), q_a))
    out = eloga' * probvec(m_out)
    return Categorical(normalize!(out, 1); check_args = false)
end

# --------------- Rules for 3 interfaces (PointMass q_a) ---------------
@rule DiscreteTransition(:out, Marginalisation) (m_in::DiscreteNonParametric, m_T1::DiscreteNonParametric, q_a::PointMass{<:AbstractArray{T, 3}}, meta::Any) where {T} = begin
    N = eltype(probvec(m_in))
    eloga = clamp.(mean(q_a), tiny(N), one(N))
    @tullio out[i] := eloga[i, a, b] * probvec(m_in)[a] * probvec(m_T1)[b]
    return Categorical(normalize!(out, 1); check_args = false)
end

@rule DiscreteTransition(:in, Marginalisation) (m_out::DiscreteNonParametric, m_T1::DiscreteNonParametric, q_a::PointMass{<:AbstractArray{T, 3}}, meta::Any) where {T} = begin
    N = eltype(probvec(m_out))
    eloga = clamp.(mean(q_a), tiny(N), one(N))
    @tullio out[i] := eloga[a, i, b] * probvec(m_out)[a] * probvec(m_T1)[b]
    return Categorical(normalize!(out, 1); check_args = false)
end

@rule DiscreteTransition(:T1, Marginalisation) (m_out::DiscreteNonParametric, m_in::DiscreteNonParametric, q_a::PointMass{<:AbstractArray{T, 3}}, meta::Any) where {T} = begin
    N = eltype(probvec(m_out))
    eloga = clamp.(mean(q_a), tiny(N), one(N))
    @tullio out[i] := eloga[a, b, i] * probvec(m_out)[a] * probvec(m_in)[b]
    return Categorical(normalize!(out, 1); check_args = false)
end

# --------------- Rules for 3 interfaces (PointMass q_T1) ---------------
@rule DiscreteTransition(:out, Marginalisation) (m_in::DiscreteNonParametric, q_a::DirichletCollection, q_T1::PointMass{<:AbstractArray{T, 3}}, meta::Any) where {T} = begin
    eloga = mean(Base.Broadcast.BroadcastFunction(clamplog), q_a)
    @tullio intermediate[i, a] := eloga[i, a, b] * probvec(q_T1)[b]
    softmax!(intermediate)
    result = intermediate * probvec(m_in)
    return Categorical(normalize!(result, 1); check_args = false)
end

@rule DiscreteTransition(:in, Marginalisation) (m_out::DiscreteNonParametric, q_a::DirichletCollection, q_T1::PointMass{<:AbstractArray{T, 3}}, meta::Any) where {T} = begin
    eloga = mean(Base.Broadcast.BroadcastFunction(clamplog), q_a)
    @tullio intermediate[a, i] := eloga[a, i, b] * probvec(q_T1)[b]
    softmax!(intermediate)
    result = intermediate' * probvec(m_out)
    return Categorical(normalize!(result, 1); check_args = false)
end

# --------------- Rules for 3 interfaces (DirichletCollection q_a) ---------------
@rule DiscreteTransition(:out, Marginalisation) (m_in::DiscreteNonParametric, m_T1::DiscreteNonParametric, q_a::DirichletCollection, meta::Any) = begin
    eloga = softmax!(mean(Base.Broadcast.BroadcastFunction(clamplog), q_a))
    @tullio out[i] := eloga[i, a, b] * probvec(m_in)[a] * probvec(m_T1)[b]
    return Categorical(normalize!(out, 1); check_args = false)
end

@rule DiscreteTransition(:in, Marginalisation) (m_out::DiscreteNonParametric, m_T1::DiscreteNonParametric, q_a::DirichletCollection, meta::Any) = begin
    eloga = softmax!(mean(Base.Broadcast.BroadcastFunction(clamplog), q_a))
    @tullio out[i] := eloga[a, i, b] * probvec(m_out)[a] * probvec(m_T1)[b]
    return Categorical(normalize!(out, 1); check_args = false)
end

@rule DiscreteTransition(:T1, Marginalisation) (m_out::DiscreteNonParametric, m_in::DiscreteNonParametric, q_a::DirichletCollection, meta::Any) = begin
    eloga = softmax!(mean(Base.Broadcast.BroadcastFunction(clamplog), q_a))
    @tullio out[i] := eloga[a, b, i] * probvec(m_out)[a] * probvec(m_in)[b]
    return Categorical(normalize!(out, 1); check_args = false)
end

# --------------- Rules for 4 interfaces (PointMass q_a) ---------------
@rule DiscreteTransition(:out, Marginalisation) (
    m_in::DiscreteNonParametric, m_T1::DiscreteNonParametric, m_T2::DiscreteNonParametric, q_a::PointMass{<:AbstractArray{T, 4}}, meta::Any
) where {T} = begin
    N = eltype(probvec(m_in))
    eloga = clamp.(mean(q_a), tiny(N), one(N))
    @tullio out[i] := eloga[i, a, b, c] * probvec(m_in)[a] * probvec(m_T1)[b] * probvec(m_T2)[c]
    return Categorical(normalize!(out, 1); check_args = false)
end

@rule DiscreteTransition(:in, Marginalisation) (
    m_out::DiscreteNonParametric, m_T1::DiscreteNonParametric, m_T2::DiscreteNonParametric, q_a::PointMass{<:AbstractArray{T, 4}}, meta::Any
) where {T} = begin
    N = eltype(probvec(m_out))
    eloga = clamp.(mean(q_a), tiny(N), one(N))
    @tullio out[i] := eloga[a, i, b, c] * probvec(m_out)[a] * probvec(m_T1)[b] * probvec(m_T2)[c]
    return Categorical(normalize!(out, 1); check_args = false)
end

@rule DiscreteTransition(:T1, Marginalisation) (
    m_out::DiscreteNonParametric, m_in::DiscreteNonParametric, m_T2::DiscreteNonParametric, q_a::PointMass{<:AbstractArray{T, 4}}, meta::Any
) where {T} = begin
    N = eltype(probvec(m_out))
    eloga = clamp.(mean(q_a), tiny(N), one(N))
    @tullio out[i] := eloga[a, b, i, c] * probvec(m_out)[a] * probvec(m_in)[b] * probvec(m_T2)[c]
    return Categorical(normalize!(out, 1); check_args = false)
end

@rule DiscreteTransition(:T2, Marginalisation) (
    m_out::DiscreteNonParametric, m_in::DiscreteNonParametric, m_T1::DiscreteNonParametric, q_a::PointMass{<:AbstractArray{T, 4}}, meta::Any
) where {T} = begin
    N = eltype(probvec(m_out))
    eloga = clamp.(mean(q_a), tiny(N), one(N))
    @tullio out[i] := eloga[a, b, c, i] * probvec(m_out)[a] * probvec(m_in)[b] * probvec(m_T1)[c]
    return Categorical(normalize!(out, 1); check_args = false)
end

# --------------- Rules for 4 interfaces (DirichletCollection q_a) ---------------
@rule DiscreteTransition(:out, Marginalisation) (m_in::DiscreteNonParametric, m_T1::DiscreteNonParametric, m_T2::DiscreteNonParametric, q_a::DirichletCollection, meta::Any) = begin
    eloga = softmax!(mean(Base.Broadcast.BroadcastFunction(clamplog), q_a))
    @tullio out[i] := eloga[i, a, b, c] * probvec(m_in)[a] * probvec(m_T1)[b] * probvec(m_T2)[c]
    return Categorical(normalize!(out, 1); check_args = false)
end

@rule DiscreteTransition(:in, Marginalisation) (m_out::DiscreteNonParametric, m_T1::DiscreteNonParametric, m_T2::DiscreteNonParametric, q_a::DirichletCollection, meta::Any) = begin
    eloga = softmax!(mean(Base.Broadcast.BroadcastFunction(clamplog), q_a))
    @tullio out[i] := eloga[a, i, b, c] * probvec(m_out)[a] * probvec(m_T1)[b] * probvec(m_T2)[c]
    return Categorical(normalize!(out, 1); check_args = false)
end

@rule DiscreteTransition(:T1, Marginalisation) (m_out::DiscreteNonParametric, m_in::DiscreteNonParametric, m_T2::DiscreteNonParametric, q_a::DirichletCollection, meta::Any) = begin
    eloga = softmax!(mean(Base.Broadcast.BroadcastFunction(clamplog), q_a))
    @tullio out[i] := eloga[a, b, i, c] * probvec(m_out)[a] * probvec(m_in)[b] * probvec(m_T2)[c]
    return Categorical(normalize!(out, 1); check_args = false)
end

@rule DiscreteTransition(:T2, Marginalisation) (m_out::DiscreteNonParametric, m_in::DiscreteNonParametric, m_T1::DiscreteNonParametric, q_a::DirichletCollection, meta::Any) = begin
    eloga = softmax!(mean(Base.Broadcast.BroadcastFunction(clamplog), q_a))
    @tullio out[i] := eloga[a, b, c, i] * probvec(m_out)[a] * probvec(m_in)[b] * probvec(m_T1)[c]
    return Categorical(normalize!(out, 1); check_args = false)
end

# --------------- Rules for 5 interfaces (PointMass q_a) ---------------
@rule DiscreteTransition(:out, Marginalisation) (
    m_in::DiscreteNonParametric, m_T1::DiscreteNonParametric, m_T2::DiscreteNonParametric, m_T3::DiscreteNonParametric, q_a::PointMass{<:AbstractArray{T, 5}}, meta::Any
) where {T} = begin
    N = eltype(probvec(m_in))
    eloga = clamp.(mean(q_a), tiny(N), one(N))
    @tullio out[i] := eloga[i, a, b, c, d] * probvec(m_in)[a] * probvec(m_T1)[b] * probvec(m_T2)[c] * probvec(m_T3)[d]
    return Categorical(normalize!(out, 1); check_args = false)
end

@rule DiscreteTransition(:in, Marginalisation) (
    m_out::DiscreteNonParametric, m_T1::DiscreteNonParametric, m_T2::DiscreteNonParametric, m_T3::DiscreteNonParametric, q_a::PointMass{<:AbstractArray{T, 5}}, meta::Any
) where {T} = begin
    N = eltype(probvec(m_out))
    eloga = clamp.(mean(q_a), tiny(N), one(N))
    @tullio out[i] := eloga[a, i, b, c, d] * probvec(m_out)[a] * probvec(m_T1)[b] * probvec(m_T2)[c] * probvec(m_T3)[d]
    return Categorical(normalize!(out, 1); check_args = false)
end

@rule DiscreteTransition(:T1, Marginalisation) (
    m_out::DiscreteNonParametric, m_in::DiscreteNonParametric, m_T2::DiscreteNonParametric, m_T3::DiscreteNonParametric, q_a::PointMass{<:AbstractArray{T, 5}}, meta::Any
) where {T} = begin
    N = eltype(probvec(m_out))
    eloga = clamp.(mean(q_a), tiny(N), one(N))
    @tullio out[i] := eloga[a, b, i, c, d] * probvec(m_out)[a] * probvec(m_in)[b] * probvec(m_T2)[c] * probvec(m_T3)[d]
    return Categorical(normalize!(out, 1); check_args = false)
end

@rule DiscreteTransition(:T2, Marginalisation) (
    m_out::DiscreteNonParametric, m_in::DiscreteNonParametric, m_T1::DiscreteNonParametric, m_T2::DiscreteNonParametric, q_a::PointMass{<:AbstractArray{T, 5}}, meta::Any
) where {T} = begin
    N = eltype(probvec(m_out))
    eloga = clamp.(mean(q_a), tiny(N), one(N))
    @tullio out[i] := eloga[a, b, c, i, d] * probvec(m_out)[a] * probvec(m_in)[b] * probvec(m_T1)[c] * probvec(m_T2)[d]
    return Categorical(normalize!(out, 1); check_args = false)
end

@rule DiscreteTransition(:T3, Marginalisation) (
    m_out::DiscreteNonParametric, m_in::DiscreteNonParametric, m_T1::DiscreteNonParametric, m_T2::DiscreteNonParametric, q_a::PointMass{<:AbstractArray{T, 5}}, meta::Any
) where {T} = begin
    N = eltype(probvec(m_out))
    eloga = clamp.(mean(q_a), tiny(N), one(N))
    @tullio out[i] := eloga[a, b, c, d, i] * probvec(m_out)[a] * probvec(m_in)[b] * probvec(m_T1)[c] * probvec(m_T2)[d]
    return Categorical(normalize!(out, 1); check_args = false)
end

# --------------- Rules for 5 interfaces (DirichletCollection q_a) ---------------
@rule DiscreteTransition(:out, Marginalisation) (
    m_in::DiscreteNonParametric, m_T1::DiscreteNonParametric, m_T2::DiscreteNonParametric, m_T3::DiscreteNonParametric, q_a::DirichletCollection, meta::Any
) = begin
    eloga = softmax!(mean(Base.Broadcast.BroadcastFunction(clamplog), q_a))
    @tullio out[i] := eloga[i, a, b, c, d] * probvec(m_in)[a] * probvec(m_T1)[b] * probvec(m_T2)[c] * probvec(m_T3)[d]
    return Categorical(normalize!(out, 1); check_args = false)
end

@rule DiscreteTransition(:in, Marginalisation) (
    m_out::DiscreteNonParametric, m_T1::DiscreteNonParametric, m_T2::DiscreteNonParametric, m_T3::DiscreteNonParametric, q_a::DirichletCollection, meta::Any
) = begin
    eloga = softmax!(mean(Base.Broadcast.BroadcastFunction(clamplog), q_a))
    @tullio out[i] := eloga[a, i, b, c, d] * probvec(m_out)[a] * probvec(m_T1)[b] * probvec(m_T2)[c] * probvec(m_T3)[d]
    return Categorical(normalize!(out, 1); check_args = false)
end

@rule DiscreteTransition(:T1, Marginalisation) (
    m_out::DiscreteNonParametric, m_in::DiscreteNonParametric, m_T2::DiscreteNonParametric, m_T3::DiscreteNonParametric, q_a::DirichletCollection, meta::Any
) = begin
    eloga = softmax!(mean(Base.Broadcast.BroadcastFunction(clamplog), q_a))
    @tullio out[i] := eloga[a, b, i, c, d] * probvec(m_out)[a] * probvec(m_in)[b] * probvec(m_T2)[c] * probvec(m_T3)[d]
    return Categorical(normalize!(out, 1); check_args = false)
end

@rule DiscreteTransition(:T2, Marginalisation) (
    m_out::DiscreteNonParametric, m_in::DiscreteNonParametric, m_T1::DiscreteNonParametric, m_T2::DiscreteNonParametric, q_a::DirichletCollection, meta::Any
) = begin
    eloga = softmax!(mean(Base.Broadcast.BroadcastFunction(clamplog), q_a))
    @tullio out[i] := eloga[a, b, c, i, d] * probvec(m_out)[a] * probvec(m_in)[b] * probvec(m_T1)[c] * probvec(m_T2)[d]
    return Categorical(normalize!(out, 1); check_args = false)
end

@rule DiscreteTransition(:T3, Marginalisation) (
    m_out::DiscreteNonParametric, m_in::DiscreteNonParametric, m_T1::DiscreteNonParametric, m_T2::DiscreteNonParametric, q_a::DirichletCollection, meta::Any
) = begin
    eloga = softmax!(mean(Base.Broadcast.BroadcastFunction(clamplog), q_a), dims = 1)
    @tullio out[i] := eloga[a, b, c, d, i] * probvec(m_out)[a] * probvec(m_in)[b] * probvec(m_T1)[c] * probvec(m_T2)[d]
    return Categorical(normalize!(out, 1); check_args = false)
end
