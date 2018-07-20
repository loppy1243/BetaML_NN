using Flux
using Flux.Tracker
using Lazy: @>

macro :(>>>)(args...)
    quote
        x -> @> x $(map(esc, args)...)
    end
end

abstract type CO <: ModelType end

const NODES = prod(2(GRIDSIZE.-1).+1)
const SCALE = (XYMAX - XYMIN)./2GRIDSIZE

regularize(x) = reshape(x/MAX_E, GRIDSIZE, 1, :)

model(activ, N) = Chain(regularize, CellLayer(), @concat(identity..., PointLayer(activ, N))
