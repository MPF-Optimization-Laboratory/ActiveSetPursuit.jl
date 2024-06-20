
@doc raw"""
```julia
    asp_bpdn(A,B)```
    solves the basis pursuit problem.

    (BP)
    ```math 
    \min_x   \|x \|_1 
    ```
      subject to 
    ```math 
        Ax = B.
    ```
    ```julia
    asp_bpdn(A,B,λ)
    ```
    solves the basis pursuit denoise problem

    (BPDN) 
    ```math 
    \min_x  \frac{1}{2} \|Ax - B\|_2^2 + λ \|x\|_1
    ```
"""
function asp_bpdn(A, b, lambda; kwargs...)
  n = size(A, 2)
  bl = -ones(n)
  bu = +ones(n)
  tracer = bpdual(A, b, lambda, bl, bu; loglevel =0, kwargs...)
  return tracer
end  
