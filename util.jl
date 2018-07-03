julienne(A, dims) = mapslices(x -> [x], A, dims)
julienne(f, A, dims) = mapslices(x -> [f(x)], A, dims)

# Alternative: create zero array of correct size and fill in with A.
function pad(A, val=0; width=1)
    ret = A
    dims = size(A) |> collect
    off = zeros(Int, ndims(A))
    for i = 1:length(dims)
        ds = dims + off
        zs = fill(val, ds[1:i-1]..., 1, ds[i+1:end]...)
        for _ = 1:width
            ret = cat(i, zs, ret)
            ret = cat(i, ret, zs)
        end
        off[i] += 2width
    end

    ret
end
