include("BetaML_Data.jl")

module BetaML_NN

import Base.Threads: @threads
import Plots; Plots.gr()

include("util.jl")
include("consts.jl")
include("io.jl")
include("models.jl")
include("modelfuncs.jl")
include("catcnn.jl")
include("regcnn.jl")
include("other.jl")

regularize(events, points) = (permutedims(events, [2, 3, 1]), permutedims(points, [2, 1]))

function main(events, points)
    events, points = regularize(events, points)

    OtherNN.train("othernn2__N_5.bson", OtherNN.other(Flux.relu=>"relu", 0.01, 0.1, 5),
                  events, points, load=false)
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
