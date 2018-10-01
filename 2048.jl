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
function moveDown!(tk48::Matrix{T}) where T
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
        z = append!(zeros(T,siz-k),z)
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
function moveUp!(tk48::Matrix{T}) where T
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
        append!(z, zeros(T,siz-k))
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
function moveRight!(tk48::Matrix{T}) where T
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
        z = append!(zeros(T,siz-k),z)
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
function moveLeft!(tk48::Matrix{T}) where T
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
        append!(z, zeros(T,siz-k))
        moved |= !iszero(tk48[i,:] - z)
        tk48[i,:] = z
    end
    return score, moved
end


"""
Doubles every non-empty tile in game board
"""
function double!(tk48::Matrix)
    x = findall(!iszero,tk48)
    for i in x
        tk48[i] += 1
    end
    return tk48
end

"""
Returns true if game is over
"""
function gameOver(tk48::Matrix)
    win = maximum(tk48) >= 11
    moved = moveUp!(copy(tk48))[2]
    moved |= moveDown!(copy(tk48))[2]
    moved |= moveRight!(copy(tk48))[2]
    moved |= moveLeft!(copy(tk48))[2]
    return !moved || win
end

#up = 4; down = 2; right = 1; left = 3;
function play!(tk48::Matrix, pscore::Real, dir::Int; dob::Bool=false)
    f = [moveUp!, moveRight!, moveDown!, moveLeft!]
    score, moved = f[1+mod(dir,4)](tk48)
    if moved
        pscore += score+1 # legal moves get a point
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
