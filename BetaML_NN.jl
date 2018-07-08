include("BetaML_Data.jl")

module BetaML_NN

import Plots; Plots.gr()

include("util.jl")
include("consts.jl")
include("io.jl")
include("models.jl")
#include("catcnn.jl")
#include("regcnn.jl")
include("other.jl")

function main(events, points)
    OtherNN.train("othernn.bson", OtherNN.model1(), permutedims(events, [2, 3, 1]),
                  permutedims(points, [2, 1]), load=false)
end

end # module DenseNet
