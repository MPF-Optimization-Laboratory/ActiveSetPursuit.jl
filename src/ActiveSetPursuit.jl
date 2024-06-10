module ActiveSetPursuit

include("BPDual.jl")
include("helpers.jl")
include("homotopy.jl")


using .BPDual
using .homotopy

export bpdual, sparsity, newtonstep, objectives, infeasibilities
export trimx, triminf, restorefeas, htpynewlam, find_step
export as_topy

end
