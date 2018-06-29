include("BetaML_Data.jl")

module BetaML_NN

include("consts.jl")
include("util.jl")
include("io.jl")
include("models.jl")
include("loss.jl")
include("catcnn.jl")

main() = CatCNN.train("catcnn_onelayer.bson", CatCNN.model2)

end # module DenseNet
