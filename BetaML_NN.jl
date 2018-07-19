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

    JLD2.jldopen("data/params.jld2", "w") do io
        ps = Flux.params(model) .|> data
        co = JLD2.Group(io, "completely_other")
        params = JLD2.Group(co, "params")
        for (p, n) in zip(ps, ["conv", "conv_bias", "dense1", "dense1_bias", "dense2", "dense2_bias"])
            params[n] = p
        end
        hyparams = JLD2.Group(co, "hyparams")
        for (k, v) in hyperparams(model)
            hyparams[string(k)] = v
        end
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
