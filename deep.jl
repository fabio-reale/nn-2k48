# using AutoGrad
using Knet

include(Knet.dir("examples","mnist.jl"))
MNIST.loaddata()
using MNIST: xtrn, ytrn, xtst, ytst, minibatch
dtrn = minibatch(xtrn, ytrn, 100)
dtst = minibatch(xtst, ytst, 100)

x = convert(Matrix{Float64},dtrn[1][1])
y = convert(Matrix{Float64},dtrn[1][2])

feedf(w::Matrix, x::Matrix) = w*x
pwLin(w::Matrix) = map(x->max(.01*x,x),w)
∇pwLin(w::Matrix) = map(x-> x < 0 ? x = .01 : x = 1., w)
∇tanh(w::Matrix) = map(x-> 1-x*x, w)
∇tanh(w::Vector) = map(x-> 1-x*x, w)
softplus(w::Real) = log(1.+e^w)

mse(z::Vector) = 0.5*sum(abs2,z)
function mse(z::Matrix)
    l = similar(z[1,:])
    for i in 1:size(z)[2]
        l[i] = 0.5*sum(abs2, z[:,i])
    end
    return l
end

function softmax(z::Vector)
    d = maximum(z)
    out = z.-d
    out = exp.(out)
    out ./= sum(out)
    return out
end
function softmax(z::Matrix)
    out = similar(z)
    d = 0
    for i in 1:size(z)[2]
        d = maximum(z[:,i])
        out[:,i] = z[:,i].-d
        out[:,i] = exp.(out[:,i])
        out[:,i] ./= sum(out[:,i])
    end
    return out
end

function ∇loss(f::Vector, w::Vector, x::Matrix)
    aux = [x,x]
    aux[2] = ∇pwLin(f[3]).*f[5] # δ_β
    #aux[2] = f[5]
    aux[1] = w[2]'*aux[2] # ∇z(s)
    aux[1] = ∇pwLin(f[1]).*aux[1] # δ_α
    aux[2] *= f[2]'
    aux[1] *= x'
    #println(size(aux[1])," ", size(aux[2]))
    return aux
end

function trainBatch(x::Matrix, y::Matrix,lr::Float64, n::Int64)
    x = cat(1,x,ones(Float64,1,size(x)[2]))
    w = [0.5*rand(Float64,64,size(x)[1]),0.5*rand(Float64,size(y)[1],64)]
    flow = [x,x,x,x,x]
    for i in 1:n
        flow[1] = w[1]*x # α
        flow[2] = ∇pwLin(flow[1]) # z
        flow[3] = w[2]*flow[2] # β
        flow[4] = ∇pwLin(flow[3]) # y
        flow[5] = flow[4]-y # δ_β
        w .-= lr*∇loss(flow,w,x)
        #println(size(w[1])," ", size(dw[1]))
        #println(size(w[2])," ", size(dw[2]))
    end
    return w,flow
end

z = trainBatch(x,y,0.1,50)
