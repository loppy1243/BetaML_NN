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
    events, points = regularize(events, points)

    pointhist("completely_other.bson", events, points)

    dir = "plots/train/pointhist"
    !ispath(dir) && mkpath(dir)
    Plots.png(dir*"/completely_other.png")
end

function rand_pointhist(points, y, p)
    points = permutedims(points, [2, 1])

    print("Computing predictions...")
    dists = mapslices(points, 2) do point
        offset = cellpoint(pointcell(point) + (0.65 < rand() ? [0, 0] : rand(-1:1, 2)))
        pred = 3rand(2).-1.5 + cellpoint(pointcell(point))
        norm(pred - point)
    end |> x -> squeeze(x, 2)
    println(" Done.")

    print("Computing statistics...") 
    me = mean(dists)
    mo = mode(dists)
    st = std(dists, mean=me)
    m = minimum(dists)
    M = maximum(dists)
    x = count(dists .< y)/length(dists)
    q = Base.quantile(dists, p)
    println(" Done.")

    print("Generating histogram...")
    linhist = stephist(dists, legend=false, title="Random", xlims=(0, 4))
    annotate!(xrel(0.7), yrel(0.9),
              "Min -- Max: "*string(round(m, 3))*" -- "*string(round(M, 3)))
    annotate!(xrel(0.7), yrel(0.8), "Mean: "*string(round(me, 3)))
    annotate!(xrel(0.7), yrel(0.7), "Mode: "*string(round(mo, 3)))
    annotate!(xrel(0.7), yrel(0.6), "STD: "*string(round(st, 3)))
    println(" Done.")

    println("(P(<3), 90th-%tile) = ", (x, q))

    (linhist, x, q)
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
