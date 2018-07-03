include("BetaML_Data.jl")

module BetaML_NN

import Plots

export @modelfunc
macro modelfunc(funcdecl)
    @assert funcdecl.head == :function || funcdecl.head == :(=) #=
         =# && funcdecl.args[1].head == :call

    fname = esc(funcdecl.args[1].args[1])

    quote
        $fname(modelfile::AbstractString, events, cells; model_name="") =
            $fname(BetaML_NN.load(modelfile)[:model], events, cells, model_name=model_name)
        $fname(models, events, cells; model_names=fill("", length(model_pairs))) =
            cells((m, n) -> $fname(m, events, cells, model_name=n), models, model_names)

        $(esc(funcdecl))
    end
end

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

    onelayer = CatCNN.accuracies("catcnn_onelayer.bson", events, cells, model_name="One Layer CNN")
    twolayer = CatCNN.accuracies("catcnn_twolayer.bson", events, cells, model_name="Two Layer CNN")

    println("One Layer CNN: ", onelayer[1])
    println("Two Layer CNN: ", twolayer[1])
    !ispath("plots/validation/acc_hists") && mkpath("plots/validation/acc_hists")
    Plots.png(onelayer[2], "plots/validation/acc_hists/catcnn_onelayer.png")
    Plots.png(twolayer[2], "plots/validation/acc_hists/catcnn_twolayer.png")
end
end # module DenseNet
