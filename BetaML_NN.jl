include("BetaML_Data.jl")

module BetaML_NN

import Plots

include("consts.jl")
include("util.jl")
include("io.jl")
include("models.jl")
include("loss.jl")
include("catcnn.jl")

main() = main(readdata(VALID_RANGE)...)
function main(events, points)
    Plots.gr()

    print("Converting points -> cells...")
    cells = mapslices(pointcell, points, 2)
    println(" Done.")

    onelayer = CatCNN.validate("catcnn_onelayer.bson", events, cells, model_name="One Layer CNN")
    twolayer = CatCNN.validate("catcnn_twolayer.bson", events, cells, model_name="Two Layer CNN")

    !ispath("plots/validation/hists") && mkpath("plots/validation/hists")
    Plots.png(onelayer, "plots/validation/hists/catcnn_onelayer.png")
    Plots.png(twolayer, "plots/validation/hists/catcnn_twolayer.png")
end

end # module DenseNet
