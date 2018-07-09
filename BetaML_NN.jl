include("BetaML_Data.jl")

module BetaML_NN

import Base.Threads: @threads
import Plots; Plots.gr()

include("util.jl")
include("consts.jl")
include("io.jl")
include("models.jl")
#include("catcnn.jl")
#include("regcnn.jl")
include("other.jl")

main() = main(BetaML_NN.readdata(TRAIN_RANGE)...)
function main(events, points)
    OtherNN.train("othernn.bson", OtherNN.model2(), permutedims(events, [2, 3, 1]),
                  permutedims(points, [2, 1]), load=true, train_dist=false)
end

# Set JULIA_NUM_THREADS
function main2(events, points)
    events = permutedims(events, [2, 3, 1])
    points = permutedims(points, [2, 1])

    M = trunc(100^(1/3)) |> Int
    i = 0
    println("Hy'param grid size: ", M^3)
    @threads for ϵ in logspace(1e-5, 1, M), η in logspace(1e-5, 1, M), N = 5:5:5M
        i += 1
        OtherNN.train("othernn_$i.bson", OtherNN.other2(relu=>"relu", ϵ, η, N), events, points)
    end
end

end # module DenseNet
