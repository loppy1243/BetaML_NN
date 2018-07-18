include("BetaML_Data.jl")

module BetaML_NN

#import Base.Threads: @threads
import Plots; Plots.gr()
import JLD
using FileIO

include("util.jl")
include("consts.jl")
include("io.jl")
include("models.jl")
include("modelfuncs.jl")
#include("catcnn.jl")
#include("regcnn.jl")
include("other.jl")

regularize(events, points) = (permutedims(events, [2, 3, 1]), permutedims(points, [2, 1]))

function main()
    model = BetaML_NN.load("models/completely_other.bson")

    for group in ["train", "valid"]
        pointhist([model, RandModel], 3.0, 0.9,
                  key=["$group/completely_other", "$group/rand"],
                  model_name=["CompletelyOther", "Random"], color=[:red, :blue],
                  size=(1200, 800))
        Plots.png("plots/$group/pointhist/co_v_rand.png")
    end
end

abstract type RandModel end
abstract type RandNModel{S} end

normal_dist(s) = () -> s*randn(2)
uniform_dist() = 3rand(2).-1.5
randmodel(events, dist) = mapslices(events, 1:2) do event
    cell = ind2sub(event, indmax(event)) |> collect
    dist() + cellpoint(cell)
end |> @λ squeeze(_, 2)
BetaML_NN.predict(::Type{RandModel}, events) = randmodel(events, uniform_dist)
BetaML_NN.predict(::Type{RandNModel{S}}, events) where S = randmodel(events, normal_dist(S))

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
