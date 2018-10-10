"""
    otherLeft(game)

Returns a matrix where all non zero elements are pushes to the left.
"""
function otherLeft(tk48::Matrix{T}) where T
    aux = similar(tk48)
    siz = size(aux,1)
    for i in 1:siz
        z = filter(!iszero, tk48[i,:])
        append!(z, zeros(T,siz-length(z)))
        aux[i,:] = z
    end
    return aux
end
"""
    otherUp(game)

Returns a matrix where all non zero elements are pushes to the top.
"""
function otherUp(tk48::Matrix{T}) where T
    aux = similar(tk48)
    siz = size(aux,2)
    for j in 1:siz
        z = filter(!iszero, tk48[:,j])
        append!(z, zeros(T,siz-length(z)))
        aux[:,j] = z
    end
    return aux
end

"""
    delta(game) -> lines_Δ, columns_Δ

returns two matrices, first contains the deltas with respect to lines and
second the deltas with respect to columns
"""
function delta(tk48::Matrix{T}) where T
    lines, columns = size(tk48)
    lin = Matrix{T}(undef, lines, columns-1)
    aux = otherLeft(tk48)
    for j in 1:columns-1
        lin[:,j] = aux[:,j]-aux[:,j+1]
    end
    col = Matrix{T}(undef, lines-1, columns)
    aux = otherUp(tk48)
    for i in 1:lines-1
        col[i,:] = aux[i,:]-aux[i+1,:]
    end
    return lin,col
end
"""
    featureDelta(game)

Returns absolute sum of deltas for game board
"""
function featureDelta(tk48::Matrix)
    l, col = delta(tk48)
    return sum(abs, l) + sum(abs, c)
end
"""
    featureZeros(game) -> Int

Returns the count of zeros on game board
"""
featureZeros(x::Matrix) = count(iszero, x)
"""
    featureNotZeros(game) -> Int

Returns the count of non-zeros on game board
"""
featureNotZeros(x::Matrix) = count(!iszero, x)
"""
    featureMaxCornerAmp(game) -> Float64

Returns amplitude from max tile in corner and max tile in game board
"""
function featureMaxCornerAmp(x::Matrix)
    k1 = maximum([x[1,1],x[1,end],x[end,1],x[end,end]])
    k = maximum(x)
    return float(k-k1)
end
"""
    featureAmp(game [, b = false]) -> Float or Tuple

Returns matrices with amplitudes by line and column.
If b = true returns the sum of all amplitudes.
"""
function featureAmp(x::Matrix; b::Bool=false)
    l = Float64[]
    for i in 1:size(x)[1]
        if !iszero(x[i,:])
            push!(l, maximum(x[i,:])-minimum(filter(!iszero,x[i,:])))
        else
            push!(l, 0.0)
        end
    end
    c = Float64[]
    for j in 1:size(x)[2]
        if !iszero(x[:,j])
            push!(c, maximum(x[:,j])-minimum(filter(!iszero,x[:,j])))
        else
            push!(c, 0.0)
        end
    end
    if b
        return sum(l)+sum(c)
    else
        return l,c
    end
end

#=
signflip is related to the delta feature, idea is to acount for increasing
AND decreasing values per line or column. The reasoning is that monotonic
boards are best boards
=#
signfliptest = :(t=[2 1 1 1; 4 2 4 3; 2 2 4 1; 2 1 4 4]) # pra testar signflip
function countFlips(v::Vector)
    if iszero(v)
        return 0
    else
        s = 1
        for i in 2:length(v)
            s += isequal(sign(v[i-1]), sign(-v[i]))
        end
        return s
    end
end
function featureSignFlip(del::NTuple)
    l = size(del[1])[1]
    c = size(del[2])[2]
    linFlip = [countFlips(del[1][i,:]) for i in 1:l]
    colFlip = [countFlips(del[2][:,j]) for j in 1:l]
    return linFlip,colFlip
end
featureSignFlip(x::Matrix) = sum(sum.(featureSignFlip(delta(x))))

function featureSum(x::Matrix)
    s = featureNotZeros(x)
    s+= featureSignFlip(x)
    s+= 10.0-featureMaxCornerAmp(x)
    s+= featureAmp(x,b=true) # alterei de forma que não retorna número
    s+= featureDelta(x)
    return s
end



function naiveStrats(tk48::Matrix,seed::Int=1,dir::Int=1)
    if  dir != 1
        dir = -1
    end
    p = collect(seed:dir:seed+(3*dir))
    for i in p
        if !iszero(play!(copy(tk48),0,i))
            return i
        end
    end
    return nothing
end

function greedyStrats(tk48::Matrix)
    aux = 0
    max = 0
    max_ind = 0
    for i in 1:4
        aux = play!(copy(tk48),0,i)
        if aux > max
            max = aux
            max_ind = i
        end
    end
    iszero(max_ind) ? return nothing : return max_ind
end

function greedyFeatureStrats(tk48::Matrix)
    min = Inf
    min_ind = 0
    for i in 1:4
        x = copy(tk48)
        iszero(play!(x,0,i)) ? aux = Inf : aux = featureSum(x)
        if aux < min
            min = aux
            min_ind = i
        end
    end
    iszero(min_ind) ? return nothing : return min_ind
end

function greedyNetStrats(tk48::Matrix, w::Vector)
    x = copy(tk48)
    x = reshape(x,prod(size(tk48)),1)
    x *= 0.05
    y = feedf(w,x)[end]
    y *= 20.0
    y = round.(y)
    return y
end

function greedyFeatureNetStrats(tk48::Matrix,w::Vector,γ::Float64)
    x,y = createFeatureBatch(tk48,w,γ)
    return findmax(y)[2]
end

function safeFeatureNetStrats(tk48::Matrix,w::Vector,γ::Float64,pscore::Real)
    x,y = createFeatureBatch(tk48,w,γ)
    #println("y = ",y,"\n")
    k = findmax(y)[2]
    while iszero(play!(copy(tk48),0,k))
        y[k] = -Inf
        k = findmax(y)[2]
        pscore-= 1.0
    end
    return pscore,k
end

function greedyOneHotNetStrats(tk48::Matrix,w::Vector,γ::Float64)
    x,y = oneHotBatch(tk48,w,γ)
    return findmax(y)[2]
end

function safeOneHotNetStrats(tk48::Matrix,w::Vector,γ::Float64,pscore::Real)
    x,y = oneHotBatch(tk48,w,γ)
    k = findmax(y)[2]
    while iszero(play!(copy(tk48),0,k))
        y[k] = -Inf
        k = findmax(y)[2]
        pscore-= 1
        println("y = ",y)
    end
    #println("\n")
    return pscore,k
end

# ok if play!returns nothing?
naivePlay!(tk48::Matrix,pscore::Real;seed::Int64=1,dir::Int64=1,double::Bool=false) = play!(tk48,pscore,naiveStrats(tk48,seed,dir),dob=double)
greedyPlay!(tk48::Matrix,pscore::Real;double::Bool=false) = play!(tk48,pscore,greedyStrats(tk48),dob=double)
greedyFeaturePlay!(tk48::Matrix,pscore::Real;double::Bool=false) = play!(tk48,pscore,greedyFeatureStrats(tk48),dob=double)

greedyFeatureNetPlay!(tk48::Matrix,pscore::Real,w::Vector,γ::Float64) = play!(tk48,pscore,greedyFeatureNetStrats(tk48,w,γ))
safeFeatureNetPlay!(tk48::Matrix,pscore::Real,w::Vector,γ::Float64) = play!(tk48,safeFeatureNetStrats(tk48,w,γ,pscore)...)

greedyOneHotNetPlay!(tk48::Matrix,pscore::Real,w::Vector,γ::Float64) = play!(tk48,pscore,greedyOneHotNetStrats(tk48,w,γ))
safeOneHotNetPlay!(tk48::Matrix,pscore::Real,w::Vector,γ::Float64) = play!(tk48,safeOneHotNetStrats(tk48,w,γ,pscore)...)

function greedyNetPlay!(tk48::Matrix,pscore::Real,w::Vector)
    auxscore = pscore
    y = greedyNetStrats(tk48,w)
    while auxscore == pscore
        i = findmax(y)[2]
        auxscore = play!(tk48,auxscore,i)
        y[i] -= 1
    end
    return auxscore
end

function playStrategy(f)
    G, score = newGame()
    while !gameOver(G)
        score = f(G, score)
    end
    printM(G,"\nscore: $(score)\n")
    #return G, score
end
