export ProdPreserveParametrisation, ProdBestSuitableParametrisation
export default_prod_strategy

import Base: prod

struct ProdPreserveParametrisation end
struct ProdBestSuitableParametrisation end

default_prod_strategy() = ProdPreserveParametrisation()