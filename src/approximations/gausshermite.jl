export ghcubature

import FastGaussQuadrature: gausshermite
import LinearAlgebra: mul!, axpy!
import PDMats: AbstractPDMat

using Distributions

const product  = Iterators.product
const repeated = Iterators.repeated
const sqrtPI1  = sqrt(pi)

const precomputed_sigma_pointsweights = (
    ([0.0], [1.7724538509055159]),
    ([-0.7071067811865476, 0.7071067811865476], [0.8862269254527579, 0.8862269254527579]),
    ([-1.2247448713915892, -8.881784197001252e-16, 1.2247448713915892], [0.29540897515091974, 1.181635900603676, 0.29540897515091974]),
    ([-1.6506801238857844, -0.5246476232752919, 0.5246476232752919, 1.6506801238857844], [0.08131283544724509, 0.8049140900055128, 0.8049140900055128, 0.08131283544724509]),
    ([-2.0201828704560856, -0.9585724646138196, -8.881784197001252e-16, 0.9585724646138196, 2.0201828704560856], [0.019953242059045872, 0.3936193231522407, 0.9453087204829428, 0.3936193231522407, 0.019953242059045872]),
    ([-2.3506049736744923, -1.3358490740136966, -0.4360774119276163, 0.4360774119276163, 1.3358490740136966, 2.3506049736744923], [0.004530009905508837, 0.1570673203228574, 0.7246295952243919, 0.7246295952243919, 0.1570673203228574, 0.004530009905508837]),
    ([-2.651961356835234, -1.6735516287674723, -0.8162878828589673, -1.3322676295501878e-15, 0.8162878828589673, 1.6735516287674723, 2.651961356835234], [0.0009717812450995172, 0.05451558281912687, 0.425607252610127, 0.8102646175568091, 0.425607252610127, 0.05451558281912687, 0.0009717812450995172]),
    ([-2.930637420257244, -1.9816567566958432, -1.1571937124467813, -0.3811869902073237, 0.3811869902073237, 1.1571937124467813, 1.9816567566958432, 2.930637420257244], [0.00019960407221136656, 0.017077983007413502, 0.20780232581489164, 0.6611470125582414, 0.6611470125582414, 0.20780232581489164, 0.017077983007413502, 0.00019960407221136656]),
    ([-3.1909932017815277, -2.2665805845318423, -1.4685532892166688, -0.7235510187528384, -0.0, 0.7235510187528384, 1.4685532892166688, 2.2665805845318423, 3.1909932017815277], [3.960697726326416e-5, 0.00494362427553695, 0.08847452739437697, 0.43265155900255486, 0.720235215606052, 0.43265155900255486, 0.08847452739437697, 0.00494362427553695, 3.960697726326416e-5]),
    ([-3.4361591188377374, -2.5327316742327897, -1.7566836492998803, -1.0366108297895167, -0.34290132722370714, 0.34290132722370714, 1.0366108297895167, 1.7566836492998803, 2.5327316742327897, 3.4361591188377374], [7.64043285523261e-6, 0.0013436457467812292, 0.03387439445548134, 0.240138611082315, 0.6108626337353252, 0.6108626337353252, 0.240138611082315, 0.03387439445548134, 0.0013436457467812292, 7.64043285523261e-6]),
    ([-3.6684708465595826, -2.7832900997816523, -2.0259480158257546, -1.3265570844949321, -0.6568095668820999, -8.881784197001252e-16, 0.6568095668820999, 1.3265570844949321, 2.0259480158257546, 2.7832900997816523, 3.6684708465595826], [1.4395603937142546e-6, 0.0003468194663233435, 0.011911395444911536, 0.11722787516770927, 0.4293597523561229, 0.6547592869145943, 0.4293597523561229, 0.11722787516770927, 0.011911395444911536, 0.0003468194663233435, 1.4395603937142546e-6]),
    ([-3.889724897869782, -3.02063702512089, -2.2795070805010615, -1.5976826351526063, -0.9477883912401648, -0.31424037625436085, 0.31424037625436085, 0.9477883912401648, 1.5976826351526063, 2.2795070805010615, 3.02063702512089, 3.889724897869782], [2.6585516843563166e-7, 8.57368704358786e-5, 0.003905390584629039, 0.05160798561588407, 0.26049231026416125, 0.5701352362624793, 0.5701352362624793, 0.26049231026416125, 0.05160798561588407, 0.003905390584629039, 8.57368704358786e-5, 2.6585516843563166e-7]),
    ([-4.10133759617864, -3.2466089783724112, -2.519735685678238, -1.853107651601512, -1.2200550365907499, -0.6057638791710609, -8.881784197001252e-16, 0.6057638791710609, 1.2200550365907499, 1.853107651601512, 2.519735685678238, 3.2466089783724112, 4.10133759617864], [4.8257318500731946e-8, 2.0430360402707125e-5, 0.0012074599927193955, 0.020862775296170085, 0.14032332068702416, 0.4216162968985409, 0.604393187921164, 0.4216162968985409, 0.14032332068702416, 0.020862775296170085, 0.0012074599927193955, 2.0430360402707125e-5, 4.8257318500731946e-8]),
    ([-4.304448570473632, -3.4626569336022714, -2.748470724985405, -2.0951832585077192, -1.4766827311411435, -0.8787137873294024, -0.2917455106725644, 0.2917455106725644, 0.8787137873294024, 1.4766827311411435, 2.0951832585077192, 2.748470724985405, 3.4626569336022714, 4.304448570473632], [8.628591168125288e-9, 4.716484355018976e-6, 0.00035509261355192335, 0.007850054726457969, 0.06850553422346574, 0.2731056090642477, 0.5364059097120885, 0.5364059097120885, 0.2731056090642477, 0.06850553422346574, 0.007850054726457969, 0.00035509261355192335, 4.716484355018976e-6, 8.628591168125288e-9]),
    ([-4.499990707309391, -3.669950373404453, -2.9671669279056054, -2.3257324861738606, -1.7199925751864926, -1.136115585210924, -0.5650695832555779, -3.552713678800501e-15, 0.5650695832555779, 1.136115585210924, 1.7199925751864926, 2.3257324861738606, 2.9671669279056054, 3.669950373404453, 4.499990707309391], [1.5224758042535364e-9, 1.059115547711077e-6, 0.00010000444123250023, 0.0027780688429127603, 0.030780033872546228, 0.1584889157959356, 0.4120286874988987, 0.5641003087264174, 0.4120286874988987, 0.1584889157959356, 0.030780033872546228, 0.0027780688429127603, 0.00010000444123250023, 1.059115547711077e-6, 1.5224758042535364e-9])
)

struct GaussHermiteCubature{PI, WI} <: AbstractApproximationMethod
    p     :: Int
    piter :: PI
    witer :: WI
end

function ghcubature(p::Int)
    points, weights = p <= length(precomputed_sigma_pointsweights) ? precomputed_sigma_pointsweights[p] : gausshermite(p)

    return GaussHermiteCubature(p, points, weights)
end

function getweights(gh::GaussHermiteCubature, mean::T, variance::T) where { T <: Real }
    return Base.Generator(gh.witer) do weight
        return weight / sqrtPI1
    end
end

function getweights(gh::GaussHermiteCubature, mean::AbstractVector{T}, covariance::AbstractPDMat{T}) where { T <: Real }
    sqrtpi = (pi ^ (length(mean) / 2))
    return Base.Generator(product(repeated(gh.witer, length(mean))...)) do pweight
        return prod(pweight) / sqrtpi
    end
end

function getpoints(gh::GaussHermiteCubature, mean::T, variance::T) where { T <: Real }
    sqrt2V = sqrt(2 * variance)
    return Base.Generator(gh.piter) do point
        return mean + sqrt2V * point
    end
end

function getpoints(cubature::GaussHermiteCubature, mean::AbstractVector{T}, covariance::AbstractPDMat{T}) where { T <: Real }
    sqrtP = sqrt(Matrix(covariance))
    sqrt2 = sqrt(2)

    tbuffer = similar(mean)
    pbuffer = similar(mean)
    return Base.Generator(product(repeated(cubature.piter, length(mean))...)) do ptuple
        copyto!(pbuffer, ptuple)
        copyto!(tbuffer, mean)
        return mul!(tbuffer, sqrtP, pbuffer, sqrt2, 1.0) # point = m + sqrt2 * sqrtP * p
    end
end

function approximate_meancov(gh::GaussHermiteCubature, g::Function, distribution)
    return approximate_meancov(gh, g, mean(distribution), cov(distribution))
end

function approximate_meancov(gh::GaussHermiteCubature, g::Function, m::T, v::T) where { T <: Real }
    weights = getweights(gh, m, v)
    points  = getpoints(gh, m, v)

    cs   = Vector{eltype(m)}(undef, length(weights))
    norm = 0.0
    mean = 0.0

    for (index, (weight, point)) in enumerate(zip(weights, points))
        gv = g(point)
        cv = weight * gv

        mean += point * cv
        norm += cv

        @inbounds cs[index] = cv
    end

    mean /= norm

    var = 0.0
    for (index, (point, c)) in enumerate(zip(points, cs))
        point -= mean
        var += c * point ^ 2
    end

    var /= norm

    return mean, var
end

function approximate_meancov(cubature::GaussHermiteCubature, g::Function, m::AbstractVector{T}, P::AbstractPDMat{T}) where { T <: Real }
    ndims = length(m)

    weights = getweights(cubature, m, P)
    points  = getpoints(cubature, m, P)

    cs = similar(m, eltype(m), length(weights))
    norm = 0.0
    mean = zeros(ndims)

    for (index, (weight, point)) in enumerate(zip(weights, points))
        gv = g(point)
        cv = weight * gv

        # mean = mean + point * weight * g(point)
        broadcast!(*, point, point, cv)  # point *= cv
        broadcast!(+, mean, mean, point) # mean += point
        norm += cv

        @inbounds cs[index] = cv
    end

    broadcast!(/, mean, mean, norm)

    cov = zeros(ndims, ndims)
    foreach(enumerate(zip(points, cs))) do (index, (point, c))
        broadcast!(-, point, point, mean)                # point -= mean
        mul!(cov, point, reshape(point, (1, ndims)), c, 1.0) # cov = cov + c * (point)⋅(point)' where c = weight * g(point)
    end

    broadcast!(/, cov, cov, norm)

    return mean, cov
end

function approximate_kernel_expectation(cubature::GaussHermiteCubature, g::Function, distribution)
    return approximate_kernel_expectation(cubature, g, mean(distribution), cov(distribution))
end

function approximate_kernel_expectation(cubature::GaussHermiteCubature, g::Function, m::AbstractVector{T}, P::AbstractPDMat{T}) where { T <: Real }
    ndims = length(m)

    weights = getweights(cubature, m, P)
    points  = getpoints(cubature, m, P)

    gbar = zeros(ndims, ndims)
    foreach(zip(weights, points)) do (weight, point)
        axpy!(weight, g(point), gbar) # gbar = gbar + weight * g(point)
    end

    return gbar
end