abstract type CO <: ModelType end

const NODES = prod(2(GRIDSIZE.-1).+1)

regularize(x) = reshape(x/MAX_E, GRIDSIZE, 1, :)

distlayer() = Chain(Conv((3, 3), 1=>1, pad(1, 1), init=ones),
                    x -> reshape(x, CELLS, :),
                    softmax,
                    x -> reshape(x, GRIDSIZE..., :))

function pointlayer(N)
    denselayer = Chain(Dense(NODES, N), Dense(N, 2))

    (x, cells) -> recenter(x, cells)
end

const model = Chain(regularize, distlayer(), pointlayer(2CELLS))
