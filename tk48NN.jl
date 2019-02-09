# for testing feedf e backProp:
M = reshape( collect(1.0:12.0), 2, 3 )
w = [M, ones(Float64, 2)]
y = [13.0, 30.0]

"""
Rotate game in desired direction
"""
rotateClock(tk48::Matrix) = reverse(tk48',dims=2)
rotateCounterClock(tk48::Matrix) = reverse(tk48',dims=1)

"""
reLU, or rectifier Linear Unit. Most popular activation function for NNs
"""
reLU(x::Number) = max(0.0, x)
∇reLU(x::Number) = x >= 0.0 ? 1.0 : 0.0

"""
reLU2 is a piecewise linear aproximation to sigmoid function, with 3 segments.
reLU(x) == reLU2(x) for x <= 1.0
"""
reLU2(x::Number) = x <= 0.5 ? max(0.0, x) : min(x, 1.0)
∇reLU2(x::Number) = 0.0 <= x <= 1.0 ? 1.0 : 0.0

"""
pwLin stands for piecewise Linear. Its near zero (~0) on negative side.
reLU(x) == pwLin(x) for x > 0
"""
pwLin(x::Number) = max(0.001*x,x)
∇pwLin(x::Number) = x >= 0.0 ? 1.0 : 0.001

"""
sigmoid function. The classic perceptron activation function.
It can be thougth of as the continuous aproximation of the ladder function
"""
sigm(x::Number) = ( 1.0/(1+exp(-x)) )
∇sigm(x::Number) = x*(1-x)

"""
tanh is implemented in Base. Only ∇tanh is required
"""
∇tanh(x::Number) = 1.0 - x^2 # mandar saída pra cá

"""
mean squared error. For vectors this is L2 norm.
For matrices, it's the mean value of mse when applied to each column
"""
mse(z::Vector) = sum(abs2,z)/size(z)[1]
mse(z::Matrix) = sum(mapslices(mse,z,dims=1))/size(z)[2]


loss(w,x,y) = mse(feedf(w,x)[end]-y)

"""
    initializeWeights(v) -> [random weight matrices]

Input v specifies #perceptrons/layer.
First parameter should be the input dimension.
Each subsequent parameter specifies the number of perceptrons in the next layer.
v can be just a list of parameters, e.g. intializeWeights(10,5,3,2), a list,
e.g. intializeWeights([10,5,3,2]) or input dimension followed by parameter list,
e.g. intializeWeights(10,[5,3,2])
"""
function initializeWeights(v...)
    w = [rand(v[2],v[1]), rand(v[2])]
    for i in 2:(length(v)-1)
        temp = [rand(v[i+1],v[i]), rand(v[i+1])]
        append!(w,temp)
    end
    return w
end
initializeWeights(v::Vector) = initializeWeights(v...)
initializeWeights(inp, v) = initializeWeights(inp, v...)

# this one is weird. It looks like this requires only one output
# this seem to be
function backProp(w::Vector,x::Vector,y::Vector)
    ff = feedf(w,x)
    dw = similar(w)
    s = ff[end]-y
    for k in length(w):-2:1
        dw[k] = s.*∇reLU.(ff[k+1])
        dw[k-1] = dw[k]*ff[k-1]'
        s = w[k-1]'*dw[k]
    end
    return dw
end

function backProp(w::Vector,x::Matrix,y::Matrix)
    ff = feedf(w,x)
    dw = similar(w)
    s = (ff[end].-y)./size(ff[end])[2]
    for k in length(w):-2:1
        dw[k] = s.*∇reLU.(ff[k+1])
        dw[k-1] = dw[k]*ff[k-1]'
        s = w[k-1]'*dw[k]
        dw[k] *= ones(Float64,size(ff[k+1])[2],1)
    end
    return dw
end

# probably w is vector of matrices of weights and x is column vector input
# size(w[1]) = ( size(i+1)[2] , size(x)[1])
# size(w[i]) = ( size(i+1)[2] , size(i-2)[1])   if (i%2 == 1) && i > 1
# size(w[i]) = ( size(i-1)[1] , 1)              if (i%2 == 0)
function feedf(w::Vector,x)
    ff = [copy(x)] # first output column is x, the input
    for i in 1:2:length(w) # ff columns alternate: linear, activation
        push!(ff, w[i]*ff[i] .+ w[i+1]) # Ax+b
        push!(ff, reLU.(ff[i+1]))
    end
    return ff
end

# ment to ease testing diferent activation functions.
# Returns a feedf function where activation function is activFunc
function activateFeedF(activFunc)
    function feedf(w::Vector, x)
        ff = [copy(x)]
        for i in 1:2:length(w)
            push!(ff, w[i]*ff[i] .+ w[i+1])
            push!(ff, activFunc.(ff[i+1]))
        end
        return ff
    end
    return feedf
end

function train(w::Vector, tk48::Matrix, lr::Float64=.1)
    # preparing x, the sample vector, including normalization step
    x = copy(tk48)
    x = 12.0 .- reshape(x,prod(size(tk48)),1)
    x = 1.0/x
    # creating, calculating, then normalizing y
    y = zeros(Float64,4)
    for i in 1:4
        # figuring out correct score values per play
        y[i] = play!(copy(tk48),0,i) # ok if play!returns nothing?
    end
    y *= .01
    dw = backProp(w, x, y)
    for i in 1:length(w)
        w[i] -= lr * dw[i]
    end
    return w
end

function train(w::Vector, tk48::Vector, batch::Int64=0,lr::Float64=.1)
    # preparing x, the sample vector, including normalization step
    x,y = createBatch(tk48,batch)
    dw = backProp(w, x, y)
    for i in 1:length(w)
        w[i] -= lr * dw[i]
    end
    return w
end

function featureTrain(w::Vector, tk48::Matrix, lr::Float64,γ::Float64)
    # preparing x, the sample vector, including normalization step
    x,y = createFeatureBatch(tk48,w,γ)
    dw = backProp(w, x, y)
    for i in 1:length(w)
        w[i] -= lr * dw[i]
    end
    return w
end

function oneHotTrain(w::Vector, tk48::Matrix, lr::Float64,γ::Float64)
    # preparing x, the sample vector, including normalization step
    x,y = oneHotBatch(tk48,w,γ)
    dw = backProp(w, x, y)
    for i in 1:length(w)
        w[i] -= lr * dw[i]
    end
    return w
end

createBatch(tk48::Matrix) = createBatch([tk48],1)
function createBatch(tk48::Vector,n::Int64=0)
    m = prod(size(tk48[1]))
    l = length(tk48)
    if n < 1 || n > l
        n = l
        ind = collect(1:n)
    else
        aux = collect(1:l)
        ind = Vector{Int64}(n)
        for i in 1:n
            ind[i] = splice!(aux,rand(1:length(aux)))
        end
    end
    y = zeros(Float64,4,n)
    for j in 1:n
        for i in 1:4
            # figuring out correct score values per play
            y[i,j] = .007*play!(copy(tk48[ind[j]]),0,i) # ok if play!returns nothing?
        end
    end
    x = zeros(Float64,m,n)
    for j in 1:n
        x[:,j] = 12.0 .- reshape(copy(tk48[ind[j]]),m,1)
        x[:,j] = 1.0/x[:,j]
    end
    return x,y
end

function createFeatureVector(tk48::Matrix)
    x = Float64[]
    push!(x,featureNotZeros(tk48)/16.)
    push!(x,featureMaxCornerAmp(tk48))
    push!(x,featureDelta(tk48)./80.)
    t = delta(tk48)
    aux = featureSignFlip(t)./3.
    # deleted featurePoints, left it here for fear of dimension mismatch
    append!(x, featurePoints(t)./10.)
    append!(x, aux[1])
    append!(x, aux[2])
    aux = featureAmp(tk48)./12.
    append!(x, aux[1])
    append!(x, aux[2])
    return x
    #reshape(x,length(x),1)
end

function createFeatureBatch(tk48::Matrix,w::Vector,γ::Float64=0.1)
    x = createFeatureVector(tk48)
    y = zeros(Float64,4)
    for i in 1:4
        aux = copy(tk48)
        y[i] = 0.007*play!(aux,0,i) # ok if play!returns nothing?
        if iszero(y[i])
            y[i]-=0.1
        elseif gameOver(aux) && maximum(aux) < 11
            y[i]-=0.1
        end
        aux = createFeatureVector(aux)
        y[i]+= γ*maximum(feedf(w,aux)[end])
    end
    return x,y
end

function oneHotVector(tk48::Matrix{Int64})
    x = Float64[]
    for i in tk48
        aux = zeros(Float64,12)
        iszero(i) ? aux[12] = 1.0 : aux[i] = 1.0
        append!(x,aux)
    end
    return x
end

function oneHotBatch(tk48::Matrix,w::Vector,γ::Float64=0.)
    x = oneHotVector(tk48)
    y = zeros(Float64,4)
    for i in 1:4
        aux = copy(tk48)
        y[i] = .007*play!(aux,0,i) # ok if play!returns nothing?
        if iszero(y[i])
            y[i]-=.1
        end
        if gameOver(aux)
            maximum(aux) < 11 ? y[i] -= .1 : y[i] = 1.
        elseif γ > 0
            #=z = similar(y)
            for j in 1:4
                z[j] = .05*play!(copy(aux),0,j)
            end
            y[i]+= γ*maximum(z)=#
            y[i]+= γ*maximum(feedf(w,oneHotVector(aux))[end])
        end
    end
    return x,y
end

# Jesus! use pwd()!!!!
s = "C:\\Users\\chong\\Downloads\\Julia\\Programas\\weight.jl"
function saveWtoFile(w::Vector; file="weight", var_name="w")
    s = "C:\\Users\\chong\\Downloads\\Julia\\Programas\\"*file*".jl"
    t = "$w"
    i = search(t,'[') - 1
    t = replace(t,t[1:i], var_name*" = ")
    out = open(s,"w")
    print(out,t)
    close(out)
end
s = "C:\\Users\\chong\\Downloads\\Julia\\Programas\\weight.jl"
impWeightMatrix(s::String) = eval(parse(readstring(s)))

function createBoards(tk48::Matrix;doub::Bool=false)
    tk48n = copy(tk48)
    tk48g = copy(tk48)
    tk48gn = copy(tk48)
    tk48gf = copy(tk48)
    tk = Array{Float64,2}[]
    scoren = 0
    scoreg = 0
    scoregn = 0
    scoregf = 0
    n = false
    g = false
    gn = false
    gf = true
    while n || g || gn || gf
        if g
            scoreg = greedyPlay!(tk48g,scoreg,double=doub)
            g = !gameOver(tk48g)
        end
        if n
            scoren = naivePlay!(tk48n,scoren,double=doub)
            n = !gameOver(tk48n)
        end
        if gn
            scoregn = greedyNaivePlay!(tk48gn,scoregn,double=doub)
            gn = !gameOver(tk48gn)
        end
        if gf
            scoregf = greedyFeaturePlay!(tk48gf,scoregf,double=doub)
            gf = !gameOver(tk48gf)
        end
        x = rand(1:5)
        y = [tk48n,tk48g,tk48gn,tk48gf]
        z = find([n,g,gn,gf])
        if x == 1 && !isempty(z)
            push!(tk,copy(y[rand(z)]))
        end
    end
    return tk
end

function randomizeBoards!(b::Vector)
    shuffle!(b)
    for j in 1:length(b)
        i = rand(1:8)
        if i==1; b[j] = b[j]'; end
        if i==2; b[j] = rotateClock(b[j]); end
        if i==3; b[j] = rotateCounterClock(b[j]); end
        if i==4; b[j] = rotateClock(rotateClock(b[j])); end
        if i==5; b[j] = flipdim(b[j],1); end
        if i==6; b[j] = flipdim(b[j],2); end
        if i==7; b[j] = rotateClock(rotateClock(b[j]))'; end
    end
end

function maxW(w::Vector)
    m = zeros(eltype(w[1]),length(w))
    for i in 1:length(w)
        m[i] = maximum(abs.(w[1]))
    end
    return maximum(m)
end

function featureAccuracy(tk48::Vector,w::Vector,γ::Float64,r::Int64)
    m = Float64[]
    for i in r:length(tk48)
        x,y = createFeatureBatch(tk48[i],w,γ)
        z = feedf(w,x)[end]
        push!(m,mse(z-y))
    end
    k = sum(m)./length(m)
    println("mse = ",k)
    return k
end


#tk48 = [0 0 0; 1 2 1; 2 3 3]
#This section is for the one hot learner

#=





tk48 = newGame()[1]
x = oneHotVector(tk48)
w = [(1./200)*rand(Float64,175,length(x)),zeros(Float64,175),
    (1./175)*rand(Float64,125,175),zeros(Float64,125),
    (1./125)*rand(Float64,112,125),zeros(Float64,112),
    (1./112)*rand(Float64,100,112),zeros(Float64,100),
    (1./100)*rand(Float64,84,100),zeros(Float64,84),
    (1./84)*rand(Float64,55,84),zeros(Float64,55),
    (1./55)*rand(Float64,42,55),zeros(Float64,42),
    (1./42)*rand(Float64,20,42),zeros(Float64,20),
    (1./20)*rand(Float64,16,20),zeros(Float64,16),
    (1./8)*rand(Float64,4,16), zeros(Float64,4)]
scores = Int64[]
maxs = Int64[]
n = 1000
training = true
eps = 1
for j in n:-1:1
    γ = 0.5
    lr= 1.
    tk48,score = newGame()
    while !gameOver(tk48)
        #printM(tk48,"score: "*string(score))
        if training
            w = oneHotTrain(w,tk48,lr,γ)
        end
        k = rand(1:20)
        if k == eps
            score = play!(tk48,score,rand(1:4)) # ok if play!returns nothing?
        else
            score = safeOneHotNetPlay!(tk48,score,w,γ)
        end
    end
    push!(scores, score)
    push!(maxs, maximum(tk48))
    if !training && maxs[end] > 10
        printM(tk48,"score: "*string(score))
    end
    println("itr: ",j,", max: ",maxs[end],", score: ", scores[end])
end
α = collect(1:length(scores))
X = [dot(α,α) sum(α);sum(α) length(α)]
Y = [dot(α,scores), sum(scores)]
Z = X\Y
println("lin.reg: ",Z[1],", max = ", maximum(maxs))



aux = scores[div(length(scores),2): end]
scores = aux
aux = maxs[div(length(maxs),2): end]
maxs = aux

#this section is for the featureLearner
tk48 = createBoards(newGame(Float64,4)[1],doub=false)
for i in 1:3
    append!(tk48, createBoards(newGame(Float64,4)[1],doub=true))
end
shuffle!(tk48)
x = createFeatureVector(tk48[1])
w = [(1./80)*rand(Float64,40,length(x)),zeros(Float64,40),(1./80)*rand(Float64,40,40),zeros(Float64,40),(1./80)*rand(Float64,40,40),zeros(Float64,40),(1./40)*rand(Float64,length(x),40),zeros(Float64,length(x)),(1./8)*rand(Float64,4,length(x)), zeros(Float64,4)]
n = 2*length(tk48)
γ = 0.2
lr= 0.1
r = div(2*length(tk48),3)
aux = -1.
ac = 0.
while aux < ac
    aux = ac
    for i in 1:n
        w = featureTrain(w,tk48[rand(1:r)],lr,γ)
    end
    ac = featureAccuracy(tk48,w,γ,r+1)
end
impr = Float64[]
for j in 20:-1:1
    n = 50
    γ = 0.2
    lr= 0.1
    for i in 1:n
        tk48,score = newGame(Float64,4)
        while !gameOver(tk48)
            w = featureTrain(w,tk48,lr,γ)
            k = rand(1:40)
            if k == 1
                score = play!(tk48,score,rand(1:4)) # ok if play!returns nothing?
            else
                score = greedyFeatureNetPlay!(tk48,score,w,γ)
            end
        end
    end

    n = 20
    scores = Vector{Int64}(n)
    maxs = Vector{Int64}(n)
    for i in 1:n
        debug=1
        tk48,score = newGame(Int64)
        while !gameOver(tk48) && debug>0
            aux = safeFeatureNetPlay!(tk48,score,w,γ)
            iszero(aux-score) ? debug = 0 : score = aux
            #printM(tk48,"score: "*string(score))
        end
        #printM(tk48,"score: "*string(score))
        scores[i] = score*debug
        maxs[i] = maximum(tk48)
    end
    #println(scores)
    #println(maxs)
    push!(impr, sum(scores)/length(scores))
    println("itr: ",j,", min: ",minimum(maxs),", max: ",maximum(maxs),", avg. score: ", impr[end])
end
α = collect(1:length(impr))
X = [dot(α,α) sum(α);sum(α) length(α)]
Y = [dot(α,impr), sum(impr)]
Z = X\Y
println("lin.reg: ",Z[1])

# this section is for the first project idea to run
tk48 = [tk48]
tk48 = createBoards(newGame(Float64,4)[1],doub=false)
#=while length(tk48) < 100
    append!(tk48, createBoards(rand(tk48),doub=false))
end=#
#randomizeBoards!(tk48)
shuffle!(tk48)
w = [(1./64)*rand(Float64,64,20),zeros(Float64,64,1),(1./32)*rand(Float64,32,64),zeros(Float64,32,1),(1./16)*rand(Float64,16,32),zeros(Float64,16,1),(1./4)*rand(Float64,4,16),zeros(Float64,4,1)]
n = 1000
k = length(tk48)
k = div(k,2)
lr = 0.9
gam = 0.999
for i in 1:n
    lr *= gam
    w = train(w,tk48,k,lr)
    if i%50 == 0
        x,y = createBatch(tk48,length(tk48))
        z = feedf(w,x)
        println("mse = ",mse(z[end]-y))
        println("maxD = ",maximum(abs.(20*(z[end]-y))))
        println("maxW = ",maxW(w),"\n")
        #=zer = 0
        for j in w
            zer += sum(iszero,j)
        end
        println("zer = ",zer,"\n")=#
    end
end




=#
