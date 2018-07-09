include("BetaML_Data.jl")

module BetaML_NN

import Base.Threads: @threads
import Plots; Plots.gr()

include("util.jl")
include("consts.jl")
include("io.jl")
include("models.jl")
include("modelfuncs.jl")
#include("catcnn.jl")
#include("regcnn.jl")
include("other.jl")

regularize(events, points) = (permutedims(events, [2, 3, 1]), permutedims(points, [2, 1]))

function main(events, points)
    path = "plots/validate/pointhist"
    events, points = regularize(events, points)

    model = BetaML_NN.load("othernn.bson")
    BetaML_NN.pointhist(model, events, points, model_name="Other2NN")

    !ispath(path) && mkpath(path)
    Plots.png(path*"/othernn.png")
end

## Set JULIA_NUM_THREADS
#function main2(events, points)
#    events, points = regularize(events, points)
#
#    M = trunc(100^(1/3)) |> Int
#    i = 0
#    println("Hy'param grid size: ", M^3)
#    @threads for ϵ in logspace(1e-5, 1, M), η in logspace(1e-5, 1, M), N = 5:5:5M
#        i += 1
#        OtherNN.train("othernn_$i.bson", OtherNN.other2(relu=>"relu", ϵ, η, N), events, points)
#    end
#end

end # module DenseNet
