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
#include("catcnn.jl")
#include("regcnn.jl")
include("other.jl")

function main(events, points) 
    Plots.gr()

    OtherNN.train("othernn.bson", OtherNN.model1(), permutedims(events, [2, 3, 1]),
                  permutedims(points, [2, 1]))
end

end # module DenseNet
