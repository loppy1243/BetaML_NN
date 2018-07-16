include("BetaML_Data.jl")

module BetaML_NN

#import Base.Threads: @threads
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
    events, points = regularize(events, points)

    model = BetaML_NN.load("completely_other.bson")
    pointhist([model, randmodel], events, points, 3.0, 0.9, model_name=["Model", "Random"],
              color=[:green, :red])

    dir = "plots/train/pointhist"
    !ispath(dir) && mkpath(dir)
    Plots.png(dir*"/completely_other_rand.png")
end

function randmodel(events)
    mapslices(events, 1:2) do event
        cell = ind2sub(event, indmax(event)) |> collect
        3rand(2).-1.5 + cellpoint(cell)
    end |> @λ squeeze(_, 2)
end
BetaML_NN.predict(::typeof(randmodel), events) = randmodel(events)

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
