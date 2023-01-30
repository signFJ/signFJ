include("graph.jl")
include("core.jl")
include("edgecore.jl")

using Laplacians
using SparseArrays

function exact(G,L,n,alpha,beta,T,k)
    LF=lap(G)
    SS=[]
    for i in T
        LF[i,i]+=beta[i]
    end
    ans=[]
    c=zeros(n)
    for i in T
        c[i]=beta[i]
    end
    o=ones(n)
    invL=inv(LF);
    for kk=1:k
        delta=zeros(n)
        kexi=invL*c;
        Lone=invL*o;
        for i=1:n
            delta[i]=alpha[i]*kexi[i]*Lone[i]/(1+alpha[i]*invL[i,i])
        end
        ss=argmax(delta)
        while ss in SS || ss in T
            delta[ss]=0
            ss=argmax(delta)
        end
        push!(SS,ss)
        push!(ans,ss)
        invL-=alpha[ss]*(invL[ss,:]*invL[ss,:]')./(1+alpha[ss]*invL[ss,ss])
    end
    return ans
end

function fast(G,LF,n,alpha,beta,T,k)
    n=G.n;
    m=G.m;
    t=Int(round(log(n)+1))
    ans=Int[];
    SS=[];
    B=getB(G)
    W=getW_half(G)
    X=spzeros(n,n)
    for i in T
        X[i,i]=beta[i]
    end
    nzs=union(T)
    sqrtX=spzeros(n,n)
    c=zeros(n)
    for i in T
        c[i]=beta[i]
    end

    o=ones(n)

    DD=spzeros(n,n)

    for kk=1:k
        delta=zeros(n);
        f=approxchol_sddm(LF+DD)
        kexi=f(c);
        Lone=f(o)
        for i in nzs
            sqrtX[i,i]=X[i,i]^0.5
        end
        Xe = zeros(n);
        Ye = zeros(n);
        B=getB(G)
        W=getW_half(G)
        for j = 1 : t
            r2 = randn(n);
            r1 = randn(m);
            xx = B'*W*r1;
            yy = sqrtX*r2;
            z1 = f(xx);
            z2 = f(yy);
            for p = 1 : n
                Xe[p] += z1[p]^2;
                Ye[p] += z2[p]^2;
            end
        end
        for i=1:n
            delta[i]=alpha[i]*kexi[i]*Lone[i]/(1+alpha[i]*(Xe[i]/t+Ye[i]/t))
        end
        ss=argmax(delta)
        while ss in SS || ss in T
            delta[ss]=0
            ss=argmax(delta)
        end
        push!(SS,ss)
        push!(ans,ss)
        X[ss]+=alpha[ss]
        push!(nzs,ss)
        DD[ss,ss]+=alpha[ss]
    end
    return ans
end


function calc(L,n,alpha,beta,S,T)
    D=spzeros(n,n)
    c=zeros(n)
    for i in S
        D[i,i]+=alpha[i]
    end
    for i in T
        D[i,i]+=beta[i]
        c[i]+=beta[i];
    end
    o=ones(n)
    f=approxchol_sddm(L+D)
    return o'*f(c)
end


function opt(G,L,n,alpha,beta,T,k)
    # LF=lap(G)
    # SS=[]
    # for i in T
    #     LF[i,i]+=beta[i]
    # end
    # ans=[]
    # c=zeros(n)
    # for i in T
    #     c[i]=beta[i]
    # end
    # o=ones(n)
    # invL=inv(LF);


    s=zeros(Int,k);
    kk=1;
    for i=1:k
        if kk in T
            kk+=1;
        end
        s[i]=kk;
        kk+=1
    end
    L=lap(G)
    tmp=calc(L,n,alpha,beta,s,T);
    indx=k;
    while s[1]<=n-k+1
        tans=calc(L,n,alpha,beta,s,T);
        if tans<tmp
            tmp=tans;
        end
        s[k]+=1;
        if s[k]>n
            s[k]-=1;
            while (indx>=1) && (s[indx]==n+indx-k)
                indx-=1;
                if indx==0
                    indx=1;
                    break;
                end
            end
            s[indx]+=1;
            for i=indx+1:k
                s[i]=s[i-1]+1;
            end
            indx=k;
        end
    end
    return tmp;

    # for kk=1:k
    #     delta=zeros(n)
    #     kexi=invL*c;
    #     Lone=invL*o;
    #     for i=1:n
    #         delta[i]=alpha[i]*kexi[i]*Lone[i]/(1+alpha[i]*invL[i,i])
    #     end
    #     ss=argmax(delta)
    #     while ss in SS || ss in T
    #         delta[ss]=0
    #         ss=argmax(delta)
    #     end
    #     push!(SS,ss)
    #     push!(ans,ss)
    #     invL-=alpha[ss]*(invL[ss,:]*invL[ss,:]')./(1+alpha[ss]*invL[ss,ss])
    # end
    # return ans
end
