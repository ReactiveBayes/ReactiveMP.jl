module ReactiveMPZygoteExt

using ReactiveMP, Zygote

import ReactiveMP: ZygoteGrad

ReactiveMP.compute_derivative(::ZygoteGrad, _, f::F, value) where {F} = Zygote.gradient(f, value)[1]
ReactiveMP.compute_gradient!(::ZygoteGrad, _, f::F, vec) where {F} = Zygote.gradient(f, vec)[1]
ReactiveMP.compute_hessian!(::ZygoteGrad, _, f::F, vec) where {F} = Zygote.hessian(f, vec)

end
