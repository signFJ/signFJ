include("graph.jl")


using LinearAlgebra
using SparseArrays
using Laplacians

fname = open("filename.txt", "r")
str   = readline(fname);
nn     = parse(Int, str);

for nnnn=1:nn
    t1=time()
    str = readline(fname);
    str = split(str);
    G = read_data(str[1])
    n=G.n;m=G.m;
    k=10;
    fout=open("1.txt", "a")
    global negedge=0;
    for i=1:G.m
        if G.w[i]<0
            global negedge+=1
        end
    end
    println(fout,str[1],' ',G.n,' ',G.m,' ',negedge,' ',negedge/G.m,' ',k);
    println(str[1],' ',G.n,' ',G.m,' ',negedge,' ',negedge/G.m,' ',k);
    t2=time()
    L=lapsp(G)
    S=getS(G)
    B=getB_all(G)
    Bp=getB_pos(G)
    Bn=getB_neg(G)
    s=(rand(G.n).-0.5).*2
    t3=time()
    # compute approx
    f=approxchol_sddm(S)
    solve=f([s; -s])
    q=zeros(G.n)
    for i=1:G.n
        q[i]=(solve[i]-solve[G.n+i])/2
    end
    aI=norm(L*q,2)^2
    aD=norm(B*q,2)^2
    aF=norm(Bp*q,2)^2
    aE=norm(Bn*q,2)^2
    aP=norm(q,2)^2/G.n
    t4=time()
    # compute exact
    dL=Matrix(L)
    for i=1:G.n
        dL[i,i]+=1
    end
    invL=inv(dL)
    eq=invL*s;
    I=norm(L*eq,2)^2
    D=norm(B*eq,2)^2
    F=norm(Bp*eq,2)^2
    E=norm(Bn*eq,2)^2
    P=norm(eq,2)^2/G.n
    t5=time()
    println(fout,"IDFEP")
    println(fout,"exact: ",I,' ',D,' ',F,' ',E,' ',P)
    println(fout,"appro: ",aI,' ',aD,' ',aF,' ',aE,' ',aP)
    println(fout,"error: ",(aI-I)/I,' ',(aD-D)/D,' ',(aF-F)/F,' ',(aE-E)/E,' ',(aP-P)/P)
    # optimize appro
    o=ones(G.n);
    sol_appro=zeros(G.n)
    sol_appro.=s;
    solveopt=f([o; -o])
    h=zeros(G.n)
    for i=1:G.n
        h[i]=(solveopt[i]-solveopt[G.n+i])/2
    end
    ini_opinion=sum(h.*s)
    ini_gain_app=zeros(k)
    ini_gain_exa=zeros(k)
    ss=zeros(G.n)
    for i=1:G.n
        ss[i]=abs(h[i])-h[i]*s[i]
    end
    for i=1:k
        tt=argmax(ss)
        ini_gain_app[i]=ss[tt]
        if h[tt]>0
            sol_appro[tt]=1
        else
            sol_appro[tt]=-1
        end
        ss[tt]=0;
    end
    t6=time()
    # optimize exact
    invL=inv(dL)
    hh=invL*o;
    sol_exa=zeros(G.n)
    sol_exa.=s;
    ss=zeros(G.n)
    for i=1:G.n
        ss[i]=abs(hh[i])-hh[i]*s[i]
    end
    for i=1:k
        tt=argmax(ss)
        ini_gain_exa[i]=ss[tt]
        if hh[tt]>0
            sol_exa[tt]=1
        else
            sol_exa[tt]=-1
        end
        ss[tt]=0;
    end
    t7=time()
    # baseline
    o=ones(G.n);
    solveopt=f([o; -o])
    h=zeros(G.n)
    for i=1:G.n
        h[i]=(solveopt[i]-solveopt[G.n+i])/2
    end
    ini_opinion=sum(h.*s)
    ini_gain_baseline_1=zeros(k)
    ini_gain_baseline_2=zeros(k)
    ini_gain_baseline_3=zeros(k)
    ini_gain_baseline_4=zeros(k)
    trust=zeros(G.n)
    for i=1:G.n
        trust[i]=L[i,i]-sum(L[i,:])
    end
    IO=zeros(G.n)
    IO.=s;
    EO=zeros(G.n)
    EO=invL*s;
    for i=1:k
        xx=rand(1:G.n)
        ini_gain_baseline_1[i]=h[xx]*(1-s[xx])
        # println(i,' ',h[xx],' ',h[xx]*(1-s[xx]))
        xx=argmax(trust)
        trust[xx]=-1;
        ini_gain_baseline_2[i]=h[xx]*(1-s[xx])
        xx=argmin(IO)
        IO[xx]=1;
        ini_gain_baseline_3[i]=h[xx]*(1-s[xx])
        xx=argmin(EO)
        EO[xx]=1;
        ini_gain_baseline_4[i]=h[xx]*(1-s[xx])
    end

    println(fout,"approx, exact,baseline1234, relative error")
    for i=1:k
        println(fout,i,' ',ini_opinion+sum(ini_gain_app[1:i]),' ',ini_opinion+sum(ini_gain_exa[1:i]),' ',ini_opinion+sum(ini_gain_baseline_1[1:i]),' ',ini_opinion+sum(ini_gain_baseline_2[1:i]),' ',ini_opinion+sum(ini_gain_baseline_3[1:i]),' ',ini_opinion+sum(ini_gain_baseline_4[1:i]),' ',1-sum(ini_gain_app[1:i])/sum(ini_gain_exa[1:i]))
    end
    println(fout,"time, init:",t2-t1,' ',t3-t2,' ',"measure approx:",t4-t3,' ',"measure exact:",t5-t4,' ',"opt approx:",t6-t5,' ',"opt exact:",t7-t6)
    close(fout)
end
close(fname)
