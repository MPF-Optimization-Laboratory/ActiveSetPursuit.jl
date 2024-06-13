function asp_homotopy(A, b; min_lambda = 0.0, 
                      homotopy = true, 
                      kwargs...)
    n = size(A, 2)
    bl = -ones(n)
    bu = +ones(n)
    active, _, xx, _, _, _, tracer = bpdual(A, b, min_lambda, bl, bu;
                             homotopy = homotopy, 
                             kwargs...)
    # BPdual's solution x is short. Make it full length.
    x = zeros(n)
    x[active] = xx
    return x, tracer
end
