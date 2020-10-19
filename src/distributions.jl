export ProdPreserveParametrisation, ProdBestSuitableParametrisation
export default_prod_strategy
export mean, median, mode, var, std, cov, invcov, entropy, pdf, logpdf

import Distributions: mean, median, mode, var, std, cov, invcov, entropy, pdf, logpdf

import Base: prod

struct ProdPreserveParametrisation end
struct ProdBestSuitableParametrisation end

default_prod_strategy() = ProdPreserveParametrisation()

prod(::ProdBestSuitableParametrisation, left, right) = prod(ProdPreserveParametrisation(), left, right)

"""
Documentation placeholder
"""
function vague end