function asp_bpdn(A, b, lambda; kwargs...)

"""
    asp_bpdn(A,B) solves the basis pursuit problem.

    (BP)   minimize_x  ||x||_1  subject to  Ax = B.

    AS_BPDN(A,B,LAM) solves the basis pursuit denoise problem

    (BPDN) minimize_x  1/2 ||Ax - B||_2^2 + LAM ||x||_1.

"""
    n = size(A, 2)
    bl = -ones(n)
    bu = +ones(n)
    tracer = bpdual(A, b, lambda, bl, bu; loglevel =0, kwargs...)

return tracer
end

