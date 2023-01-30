
function ClosenessCentrality(G)
    gg = zeros(Int, G.n)
    foreach(i -> gg[i] = i, 1 : G.n)
    g = Array{Array{Int32, 1}, 1}(undef, G.n)
    foreach(i -> g[i] = [], 1 : G.n)
    for i=1:G.m
        u=G.u[i];
        v=G.v[i];
        push!(g[u], v)
        push!(g[v], u)
    end
    C = zeros(G.n)
    d = zeros(Int32, G.n)
    Q = zeros(Int32, G.n+10)
    for s = 1 : G.n
        d .= -1
        d[s] = 0
        front = 1
        rear = 1
        Q[1] = s

        while front <= rear
            v = Q[front]
            front += 1
            for w in g[v]
                if d[w] < 0
                    rear += 1
                    Q[rear] = w
                    d[w] = d[v] + 1
                end
            end
        end

        C[s] = sum(d)
    end

    foreach(i -> C[i] = 1.0 / C[i], 1 : G.n)

    return C
end



function BetweennessCentrality(G)
    gg = zeros(Int, G.n)
    foreach(i -> gg[i] = i, 1 : G.n)
    g = Array{Array{Int32, 1}, 1}(undef, G.n)
    foreach(i -> g[i] = [], 1 : G.n)
    for i=1:G.m
        u=G.u[i];
        v=G.v[i];
        push!(g[u], v)
        push!(g[v], u)
    end
    C = zeros(G.n)
    p = Array{Array{Int32, 1}, 1}(undef, G.n)
    d = zeros(Int32, G.n)
    S = zeros(Int32, G.n+10)
    sigma = zeros(G.n)
    Q = zeros(Int32, G.n+10)
    delta = zeros(G.n)
    for s = 1 : G.n
        foreach(i -> p[i] = [], 1 : G.n)
        top = 0
        sigma .= 0
        sigma[s] = 1.0
        d .= -1
        d[s] = 0
        front = 1
        rear = 1
        Q[1] = s

        while front <= rear
            v = Q[front]
            front += 1
            top += 1
            S[top] = v
            for w in g[v]
                if d[w] < 0
                    rear += 1
                    Q[rear] = w
                    d[w] = d[v] + 1
                end
                if d[w] == (d[v] + 1)
                    sigma[w] += sigma[v]
                    push!(p[w], v)
                end
            end
        end

        delta .= 0

        while top > 0
            w = S[top]
            top -= 1
            for v in p[w]
                delta[v] += ((sigma[v] / sigma[w]) * (1 + delta[w]))
                if w != s
                    C[w] += delta[w]
                end
            end
        end

    end

    return C
end


function PageRankCentrality(G)
    n=G.n;
    a=adjsp(G)
    l=lapsp(G)
    d=zeros(n)
    for i=1:n
        d[i]=l[i,i]
    end
    p=d.^(-1)'*a
    dd=0.85;

    v=rand(n)
    v=v./norm(v,1)
    for i=1:10
        v=dd*ones(n)+(1-dd)*p*v
    end
    return v
end
