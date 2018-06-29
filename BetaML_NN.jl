module BetaML_NN

include("readdata.jl")

import BSON
using Flux
using Plots; gr()
using Flux.Optimise: train!
using Flux.Tracker: data

include("consts.jl")
include("main.jl")

const MODELFILE = "cat_pure_cnnmodel_1.bson"
include("catcnn.jl")

main() = catcnnmodel_train_main()
end # module DenseNet
