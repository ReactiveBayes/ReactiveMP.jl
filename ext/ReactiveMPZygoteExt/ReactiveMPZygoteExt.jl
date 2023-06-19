module ReactiveMPZygoteExt

using ReactiveMP, Zygote

import ReactiveMP: ZygoteGrad

ReactiveMP.compute_gradient(::ZygoteGrad, f::F, vec) where {F} = Zygote.gradient(f, vec)[1]
ReactiveMP.compute_hessian(::ZygoteGrad, f::F, vec) where {F} = Zygote.hessian(f, vec)
ReactiveMP.compute_derivative(::ZygoteGrad, f::F, value) where {F} = Zygote.gradient(f, value)[1]

end
