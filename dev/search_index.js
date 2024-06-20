var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = ActiveSetPursuit","category":"page"},{"location":"#ActiveSetPursuit","page":"Home","title":"ActiveSetPursuit","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for ActiveSetPursuit.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [ActiveSetPursuit]","category":"page"},{"location":"#ActiveSetPursuit.asp_bpdn-Tuple{Any, Any, Any}","page":"Home","title":"ActiveSetPursuit.asp_bpdn","text":"julia     asp_bpdn(A,B)     solves the basis pursuit problem.\n\n(BP)\n```math \n\\min_x   \\|x \\|_1 \n```\n  subject to \n```math \n    Ax = B.\n```\n```julia\nasp_bpdn(A,B,λ)\n```\nsolves the basis pursuit denoise problem\n\n(BPDN) \n```math \n\\min_x  \\frac{1}{2} \\|Ax - B\\|_2^2 + λ \\|x\\|_1\n```\n\n\n\n\n\n","category":"method"},{"location":"#ActiveSetPursuit.asp_omp-Tuple{Union{LinearOperators.AbstractLinearOperator, AbstractMatrix}, Vector, Real}","page":"Home","title":"ActiveSetPursuit.asp_omp","text":"```julia\nasp_omp(A,B)```\nOrthogonal matching pursuit for sparse \n```math \nAx=b\n```\nApplies the orthogonal matching pursuit (OMP) algorithm to\nestimate a sparse solution of the underdetermined system `Ax=b`.\n\n(BP)   \n```math \n\\min_x  \\|x\\|_1  \n```\nsubject to  \n```math \nAx = b.\n```\n\n\n\n\n\n","category":"method"},{"location":"#ActiveSetPursuit.bpdual-Tuple{Union{LinearOperators.AbstractLinearOperator, AbstractMatrix}, Vector, Real, Vector, Vector}","page":"Home","title":"ActiveSetPursuit.bpdual","text":"julia     function bpdual(A, b, λin, bl, bu; kwargs...) \n\nSolve the optimization problem:\n\nDP:\n```math \n    \\min_{y} \\left( -b^T y + \\frac{1}{2} λ y^T y \\right)\n```\n    subject to:\n ```math \n    bl \\leq A^T y \\leq bu\n    ```\n\nusing given `A`, `b`, `bl`, `bu`, and `λ`. When `bl = -e` and `bu = e = ones(n, 1)`,\nDP is the dual Basis Pursuit problem:\n\nBPdual:\n\n```math \n\\max_{y} \\left( b^T y - \\frac{1}{2} λ y^T y \\right)\n```\n    subject to\n    \n```math\n    \\|A^T y\\|_{\\infty} \\leq 1.\n```\n\n* Input\n* `A` : `m`-by-`n` explicit matrix or linear operator.\n* `b` : `m`-vector.\n* `λin` : Nonnegative scalar.\n* `bl`, `bu` : `n`-vectors (bl lower bound, bu upper bound).\n* `active`, `state`, `y`, `S`, `R` : May be empty or output from `BPdual` with a previous value of `λ`.\n* `loglevel` : Logging level.\n* `coldstart` : Boolean indicating if a cold start should be used.\n* `homotopy` : Boolean indicating if homotopy should be used.\n* `λmin` : Minimum value for `λ`.\n* `trim` : Number of constraints to trim.\n* `itnMax` : Maximum number of iterations.\n* `feaTol` : Feasibility tolerance.\n* `optTol` : Optimality tolerance.\n* `gapTol` : Gap tolerance.\n* `pivTol` : Pivot tolerance.\n* `actMax` : Maximum number of active constraints.\n\n* Output\n* `tracer` : A structure to store trace information at each iteration of the optimization process.\n    It contains:\n        * `active::Vector{Int}`: The indices of the active constraints.\n        * `activesoln::Vector{T}`: The solutions corresponding to the current active set.\n        * `lambda::Vector{T}`: Lambda values.\n        * `N::Int`: The total number of variables in the solution vector.\n\n\n\n\n\n","category":"method"}]
}
