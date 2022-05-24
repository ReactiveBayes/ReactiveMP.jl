export Adam, update!

mutable struct Adam{T} <: Optimizer
    x    :: T
    s    :: T
    r    :: T
    shat :: T
    rhat :: T
    ρ1   :: Float64
    ρ2   :: Float64
    λ    :: Float64
    it   :: Int64
    tmp  :: T
end

function Adam(x::Vector{Float64}; ρ1::Float64 = 0.9, ρ2::Float64 = 0.999, λ::Float64 = 1e-8)
    return Adam(
        x,
        zeros(Float64, size(x)),
        zeros(Float64, size(x)),
        zeros(Float64, size(x)),
        zeros(Float64, size(x)),
        ρ1,
        ρ2,
        λ,
        1,
        zeros(Float64, size(x))
    )
end

function Adam(x::T; ρ1::Float64 = 0.9, ρ2::Float64 = 0.999, λ::Float64 = 1e-8) where {T <: Real}
    return Adam(float(x), 0.0, 0.0, 0.0, 0.0, ρ1, ρ2, λ, 1, 0.0)
end

getx(optimizer::Adam)    = return optimizer.x
gets(optimizer::Adam)    = return optimizer.s
getr(optimizer::Adam)    = return optimizer.r
getshat(optimizer::Adam) = return optimizer.shat
getrhat(optimizer::Adam) = return optimizer.rhat
getρ1(optimizer::Adam)   = return optimizer.ρ1
getρ2(optimizer::Adam)   = return optimizer.ρ2
getλ(optimizer::Adam)    = return optimizer.λ
getit(optimizer::Adam)   = return optimizer.it
getall(optimizer::Adam)  = return optimizer.x, optimizer.s, optimizer.r, optimizer.shat, optimizer.rhat, optimizer.ρ1, optimizer.ρ2, optimizer.λ, optimizer.it, optimizer.tmp

function setx!(optimizer::Adam{T}, x::T) where {T <: Real}
    optimizer.x = x
end
function setx!(optimizer::Adam{T}, x::T) where {T <: AbstractArray}
    optimizer.x .= x
end

function sets!(optimizer::Adam{T}, s::T) where {T <: Real}
    optimizer.s = s
end
function sets!(optimizer::Adam{T}, s::T) where {T <: AbstractArray}
    optimizer.s .= s
end

function setr!(optimizer::Adam{T}, r::T) where {T <: Real}
    optimizer.r = r
end
function setr!(optimizer::Adam{T}, r::T) where {T <: AbstractArray}
    optimizer.r .= r
end

function setshat!(optimizer::Adam{T}, shat::T) where {T <: Real}
    optimizer.shat = shat
end
function setshat!(optimizer::Adam{T}, shat::T) where {T <: AbstractArray}
    optimizer.shat .= shat
end

function setrhat!(optimizer::Adam{T}, rhat::T) where {T <: Real}
    optimizer.rhat = rhat
end
function setrhat!(optimizer::Adam{T}, rhat::T) where {T <: AbstractArray}
    optimizer.rhat .= rhat
end

function setρ1!(optimizer::Adam{T}, ρ1::Float64) where {T}
    optimizer.ρ1 = ρ1
end

function setρ2!(optimizer::Adam{T}, ρ2::Float64) where {T}
    optimizer.ρ2 = ρ2
end

function setλ!(optimizer::Adam{T}, λ::Float64) where {T}
    optimizer.λ = λ
end

function setit!(optimizer::Adam{T}, it::Int64) where {T}
    optimizer.it = it
end

function update!(optimizer::Adam{T}, ∇::T) where {T <: Real}

    # fetch parameters
    x, s, r, _, _, ρ1, ρ2, λ, it, tmp = getall(optimizer)

    # update (biased) first moment
    s *= ρ1
    tmp = ∇
    tmp *= (1 - ρ1)
    s += tmp
    sets!(optimizer, s)

    # update (unbiased) first moment
    shat = s / (1 - ρ1^it)

    # update (biased) second moment
    r *= ρ2
    tmp = ∇
    tmp ^= 2
    tmp *= (1 - ρ2)
    r += tmp
    setr!(optimizer, r)

    # update (unbiased) second moment
    rhat = r / (1 - ρ2^it)

    # perform accelerated gradient step
    rhat = sqrt(rhat)
    rhat += 1e-20
    rhat /= λ
    shat /= rhat
    x -= shat
    setx!(optimizer, x)

    # update iteration count
    setit!(optimizer, it + 1)
end

function update!(optimizer::Adam{T}, ∇::T) where {T <: AbstractArray}

    # fetch parameters
    x, s, r, shat, rhat, ρ1, ρ2, λ, it, tmp = getall(optimizer)

    # update (biased) first moment
    s .*= ρ1
    tmp .= ∇
    tmp .*= (1 - ρ1)
    s .+= tmp

    # update (unbiased) first moment
    shat .= s
    shat ./= (1 - ρ1^it)

    # update (biased) second moment
    r .*= ρ2
    tmp .= ∇
    tmp .^= 2
    tmp .*= (1 - ρ2)
    r .+= tmp

    # update (unbiased) second moment
    rhat .= r
    rhat ./= (1 - ρ2^it)

    # perform accelerated gradient step
    tmp .= sqrt.(rhat)
    tmp .+= 1e-20
    tmp ./= λ
    shat ./= tmp
    x .-= shat

    # update iteration count
    setit!(optimizer, it + 1)
end
