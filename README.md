# ActiveSetPursuit

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://MPF-Optimization-Laboratory.github.io/ActiveSetPursuit.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://MPF-Optimization-Laboratory.github.io/ActiveSetPursuit.jl/dev/)
[![Build Status](https://github.com/MPF-Optimization-Laboratory/ActiveSetPursuit.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/MPF-Optimization-Laboratory/ActiveSetPursuit.jl/actions/workflows/CI.yml?query=branch%3Amain)


## ActiveSetPursuit

`ActiveSetPursuit` is a Julia package providing a framework for solving several variations of the sparse optimization problem:

$$\underset{x}{\text{minimize}} \hspace{0.5em}  \lambda \||x\||_1 + \frac{1}{2} \||Ax - b\||^2_2$$

The package is a Julia port of a similar [Matlab `asp` package](https://github.com/MPF-Optimization-Laboratory/asp). 

### Implemented Algorithms:
- Basis Pursuit Denoising
- Orthogonal Matching Pursuit
- Homotopy Basis Pursuit Denoising

### Installation:

The package can be installed as follows:
```jlcon
julia> ] add ActiveSetPursuit
```


## Example Usage

The `asp_bpdn` function within the `ActiveSetPursuit` package efficiently solves the basis pursuit denoising problem.

Ensure that the following inputs are defined before running the `asp_bpdn` function:
- **`A`**: The matrix or linear operator, size `m`-by-`n`.
- **`b`**: The vector of observations or measurements, size `m`.
- **`λin`**: A nonnegative scalar that serves as the regularization parameter.

To solve the basis pursuit denoising problem, execute the following command in Julia:

```julia
N, M = 1000, 300
A = randn(N, M)
xref = zeros(M)
xref[randperm(M)[1:20]] = randn(20)  # 20 non-zero entries
b = A * xref
const λin = 0.0
tracer = asp_bpdn(A, b, λin, traceFlag = true)
```
After the optimization process completes, if `traceFlag` was set to true, the solution vector and the regularization parameter at any iteration `itn` can be accessed as follows:


```jlcon
xx, λ = tracer[itn]
```
To extract the final iterate:

 ```jlcon
x_final, λ_final = tracer[end]
```

## Reference 

Michael Friedlander and Michael Saunders. A dual active-set quadratic programming method for finding sparse least-squares solutions, DRAFT Technical Report, Dept of Computer Science, University of British Columbia, July 30, 2012; updated April 13, 2019.  [[PDF]](https://friedlander.io/files/pdf/bpprimal.pdf)
