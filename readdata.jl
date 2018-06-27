using Base.Iterators: drop, take
using IterTools: takenth, chain
using Lazy: @>

const EVENTS = 10^6
const COLUMNS = 16*16 + 6
const ROWS = EVENTS
const GRIDSIZE = [16, 16]
const CELLS = prod(GRIDSIZE)

const XYMIN = -[48, 48]/2 # = [-24, -24]
const XYMAX = [48, 48]/2 # = [24, 24]
const XYOFF = -[48, 48]/2 # = [-24, -24]

julienne(A, dims) = squeeze(mapslices(x -> [x], A, dims), dims=dims)

function readdata(file)
    data = readdlm(file, Float32, dims=(ROWS, COLUMNS))

    (map(x -> sparse(reshape(x, GRIDSIZE...)), julienne(data[:, 1:end-6], 2)),
     julienne(data[:, end-5:end], 2))
end

function readdata(file, range::AbstractRange{T}) where T <: Integer
    @assert start(range) > 0
    buf = IOBuffer()
    open(file) do stream
        itr = @> stream begin
            eachline(chomp=false)
            drop(first(range)-1)
            i -> chain([first(i)], takenth(drop(i, 1), step(range)))
            take(length(range))
        end

        for line in itr; write(buf, line) end
    end

    seekstart(buf)
    data = readdlm(buf, Float32, dims=(length(range), COLUMNS))

    (map(x -> sparse(reshape(x, GRIDSIZE...)), julienne(data[:, 1:end-6], 2)),
     julienne(data[:, end-5:end], 2))
end
