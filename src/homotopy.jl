function asp_homotopy(A, b; min_lambda = 0.0, 
                      homotopy = true, 
                      kwargs...)
    n = size(A, 2)
    bl = -ones(n)
    bu = +ones(n)
    tracer = bpdual(A, b, min_lambda, bl, bu;
                             homotopy = homotopy, 
                             kwargs...)

    return tracer
end
