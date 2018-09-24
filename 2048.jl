"""
    printM(m [, name="Matrix:\n"])

Prints matrix m without separators and []. name is printed as header.

# Example
```
julia> printM([1 2 3; 4 5 6])
Matrix:
1 2 3
4 5 6

julia>
```
"""
function printM(x::Matrix, name::AbstractString="Matrix:\n")
    s = name*string(x)
    s = replace(s,"["=>"")
    s = replace(s,"]"=>"\n")
    s = replace(s,"; "=>"\n")
    print(s)
    #return s
end

"""
    newGame(T=Int, n=4)

Returns a board of size n x n where the types of theelements are T and 1 legal move has been made
"""
function newGame(T::Type=Int, siz::Int=4)
    tk48 = zeros(T,siz,siz)
    tk48[rand(1:(siz*siz))] = 1
    score = 0
    while iszero(score)
        score = play!(tk48,score,rand(1:4))
    end
    return tk48, convert(T,0)
end

"""
    moveDown!(game) -> move_score

Executes moving the pieces down in game board. Returns the score produced by this move.
Other moves are moveRight!, moveUp! and moveLeft!
"""
function moveDown!(tk48::Matrix)
    # score is not yet final, moved checks for any alteration
    score = 0
    moved = false
    siz = size(tk48,1)
    for j in 1:siz
        # for each column, remove zeros, then collapse equals
        z = filter!(!iszero,tk48[:,j])
        k = length(z)
        i = k
        while i > 1
            if z[i] == z[i-1]
                splice!(z,i)
                z[i-1] += 1
                score += z[i-1]
                k -= 1
                i -= 1
            end
            i -= 1
        end
        # add necessary zeros to the beginning of columns
        z = append!(zeros(eltype(tk48),siz-k),z)
        moved |= !iszero(tk48[:,j] - z)
        tk48[:,j] = z
    end
    return score, moved
end
"""
    moveUp!(game) -> move_score

Executes moving the pieces up in game board. Returns the score produced by this move.
Other moves are moveRight!, moveDown! and moveLeft!
"""
function moveUp!(tk48::Matrix)
    score = 0
    moved = false
    siz = size(tk48,1)
    for j in 1:siz
        z = filter!(!iszero,tk48[:,j])
        i = 1
        k = length(z)
        while i < k
            if z[i] == z[i+1]
                splice!(z,i)
                z[i] += 1
                score += z[i]
                k -= 1
            end
            i +=1
        end
        append!(z, zeros(eltype(tk48),siz-k))
        moved |= !iszero(tk48[:,j] - z)
        tk48[:,j] = z
    end
    return score, moved
end
"""
    moveRight!(game) -> move_score

Executes moving the pieces down in game board. Returns the score produced by this move.
Other moves are moveDown!, moveUp! and moveLeft!
"""
function moveRight!(tk48::Matrix)
    score = 0
    moved = false
    siz = size(tk48,1)
    for i in 1:siz
        z = filter!(!iszero,tk48[i,:])
        k = length(z)
        j = k
        while j > 1
            if z[j] == z[j-1]
                splice!(z,j)
                z[j-1] += 1
                score += z[j-1]
                k -= 1
                j -= 1
            end
            j -= 1
        end
        z = append!(zeros(eltype(tk48),siz-k),z)
        moved |= !iszero(tk48[i,:] - z)
        tk48[i,:] = z
    end
    return score, moved
end
"""
    moveLeft!(game) -> move_score

Executes moving the pieces down in game board. Returns the score produced by this move.
Other moves are moveDown!, moveUp! and moveRight!
"""
function moveLeft!(tk48::Matrix)
    score = 0
    moved = false
    siz = size(tk48,1)
    for i in 1:siz
        z = filter!(!iszero,tk48[i,:])
        j = 1
        k = length(z)
        while j < k
            if z[j] == z[j+1]
                splice!(z,j)
                z[j] += 1
                score += z[j]
                k -= 1
            end
            j +=1
        end
        append!(z, zeros(eltype(tk48),siz-k))
        moved |= !iszero(tk48[i,:] - z)
        tk48[i,:] = z
    end
    return score, moved
end

function otherLeft(tk48::Matrix)
    aux = copy(tk48)
    siz = size(aux,1)
    for i in 1:siz
        z = filter!(!iszero,aux[i,:])
        append!(z, zeros(eltype(aux),siz-length(z)))
        aux[i,:] = z
    end
    return aux
end

function otherUp(tk48::Matrix)
    aux = copy(tk48)
    siz = size(aux,2)
    for j in 1:siz
        z = filter!(!iszero,aux[:,j])
        append!(z, zeros(eltype(aux),siz-length(z)))
        aux[:,j] = z
    end
    return aux
end

#= delta is ment to capture how good a board is based on the difference
each tile has with its adjacent tiles =#
function delta(tk48::Matrix)
    lin = Matrix{eltype(tk48)}(undef, size(tk48).-(0,1))
    aux = otherLeft(tk48)
    for j in 1:size(lin)[2]
        lin[:,j] = aux[:,j]-aux[:,j+1]
    end
    col = Matrix{eltype(tk48)}(undef, size(tk48).-(1,0))
    aux = otherUp(tk48)
    for i in 1:size(col)[1]
        col[i,:] = aux[i,:]-aux[i+1,:]
    end
    return lin,col
end
featureDelta(tk48::Matrix) = sum(sum.(abs,delta(tk48)))

featureZeros(x::Matrix) = sum(iszero.(x))
featureNotZeros(x::Matrix) = sum(.!iszero.(x))
function featureMaxCornerAmp(x::Matrix)
    k1 = maximum([x[1,1],x[1,end],x[end,1],x[end,end]])
    k = maximum(x)
    iszero(k-k1) ? k-= 1.0 : k = 0.
    return k
end
function featureAmp(x::Matrix; b::Bool=false)
    l = Float64[]
    for i in 1:size(x)[1]
        if !iszero(x[i,:])
            push!(l, maximum(x[i,:])-minimum(filter(!iszero,x[i,:])))
        else
            push!(l,0.0)
        end
    end
    c = Float64[]
    for j in 1:size(x)[2]
        if !iszero(x[:,j])
            push!(c, maximum(x[:,j])-minimum(filter(!iszero,x[:,j])))
        else
            push!(c,0.)
        end
    end
    if b
        return sum(l)+sum(c)
    else
        return l,c
    end
end
featurePoints(del::NTuple) = count.(iszero,del)
featurePoints(x::Matrix) = 16-sum(count.(iszero,delta(x)))

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
    s+= featurePoints(x)
    s+= featureDelta(x)
    return s
end

rotateClock(tk48::Matrix) = reverse(tk48,dims=1)'
rotateCounterClock(tk48::Matrix) = reverse(tk48,dims=2)'
function double!(tk48::Matrix)
    x = findall(!iszero,tk48)
    for i in x
        tk48[i] += 1
    end
    return tk48
end

# Returns true if game is over
function gameOver(tk48::Matrix)
    win = maximum(tk48) >= 11
    moved = moveUp!(copy(tk48))[2]
    #moved |= moveDown!(copy(tk48))[2]
    moved |= moveRight!(copy(tk48))[2]
    #moved |= moveLeft!(copy(tk48))[2]
    return !moved || win
end

function play!(tk48::Matrix, pscore::Real, dir::Int64; dob::Bool=false)
    f = [moveUp!, moveDown!, moveRight!, moveLeft!]
    dir =
    score, moved = f[1+mod(dir,4)](tk48)
    if moved
        # add movement score
        pscore += score+1
        if maximum(tk48) >= 11
            pscore += 20
        end
        # randomly double entire board (used to create high end boards)
        if dob
            j = rand(1:16)
            if j == 1
                double!(tk48)
            end
        end
        # choose new tile and add its score
        i = rand(findall(iszero,tk48))
        k = rand(1:10)
        k == 2 ? tk48[i] = 2 : tk48[i] = 1
    end
    #printM(tk48," 2048")
    return pscore
end

#up = 1; down = 3; right = 2; left = 0;
function naiveStrats(tk48::Matrix,seed::Int64=1,dir::Int64=1)
    if  dir != 1
        dir = -1
    end
    p = collect(seed:dir:seed+(3*dir))
    for i in p
        if !iszero(play!(copy(tk48),0,i))
            return i
        end
    end
    return seed
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
    return max_ind
end

function greedyNaiveStrats(tk48::Matrix,seed::Int64=1,dir::Int64=1)
    if  dir != 1
        dir = -1
    end
    p = collect(seed:dir: seed+(3*dir))
    s = zeros(eltype(tk48),4)
    for i in 1:4
        s[i] = play!(copy(tk48),0,p[i])
    end
    if s[3] > s[2] > s[1]
        return p[3]
    elseif s[2] > s[1] > 0
        return p[2]
    else
        aux = findfirst(!iszero,s)
        return p[ aux == nothing ? seed : aux ]
    end
end

function greedyFeatureStrats(tk48::Matrix)
    min = Inf64
    min_ind = 0
    for i in 1:4
        x = copy(tk48)
        iszero(play!(x,0,i)) ? aux = Inf64 : aux = featureSum(x)
        if aux < min
            min = aux
            min_ind = i
        end
    end
    return min_ind
end

function greedyNetStrats(tk48::Matrix, w::Vector)
    x = copy(tk48)
    x = reshape(x,prod(size(tk48)),1)
    x *= .05
    y = feedf(w,x)[end]
    y *= 20.
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

naivePlay!(tk48::Matrix,pscore::Real;seed::Int64=1,dir::Int64=1,double::Bool=false) = play!(tk48,pscore,naiveStrats(tk48,seed,dir),dob=double)
greedyPlay!(tk48::Matrix,pscore::Real;double::Bool=false) = play!(tk48,pscore,greedyStrats(tk48),dob=double)
greedyNaivePlay!(tk48::Matrix,pscore::Real;seed::Int64=1,dir::Int64=1,double::Bool=false) = play!(tk48,pscore,greedyNaiveStrats(tk48,seed,dir),dob=double)
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
