module PointPredLayer

using Lazy: @>
using ConcatLayer

const NODES = prod(2(GRIDSIZE.-1).+1)
const SCALE = (XYMAX - XYMIN)./2GRIDSIZE

struct PointLayer{T, F<:Function}
    denselayer::T
    activ::F
end
function PointLayer(activ, N)
    l = Chain(Dense(NODES, N, activ), Dense(N, 2))

    PointLayer{typeof(l), typeof(activ)}(l, activ)
end
(p::PointLayer)(x) = @> x[1] begin
    recenter(x[2])
    p.activ.()
    vec
    p.denselayer
    (*).(SCALE)
end

function recenter(dists, cells)
    ret = zeros(eltype(dists), (2GRIDSIZE .+ 1)..., size(dists, 3))

    for k = 1:GRIDSIZE[1], l = 1:GRIDSIZE[2], i in indices(dists, 3)
        ixs = [k, l] - cells[:, i] + GRIDSIZE .+ 1

        ret[ixs..., i] = dists[k, l, i]
    end

    Tracker.collect(ret)
end

end # module PointPredLayer
