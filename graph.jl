mutable struct Graph
    n :: Int # |V|
    m :: Int # |E|
    u :: Array{Int32, 1}
    v :: Array{Int32, 1} # uv is an edge
    w :: Array{Int32, 1} # weight of each edge
    #nbr :: Array{Array{Int32, 1}, 1}
end
using LinearAlgebra
using SparseArrays
using Laplacians


function read_data(filename::AbstractString)
    ffname=string("data/",filename)
    open(ffname) do file
        G=Graph(0,0,Int[],Int[],Int[])
        for line in eachline(file)
            fields = split(line)
            if parse(Int, fields[3])>0
                push!(G.u, parse(Int, fields[1]))
                push!(G.v, parse(Int, fields[2]))
                push!(G.w, 1)
                G.m+=1
            elseif parse(Int, fields[3])<0
                push!(G.u, parse(Int, fields[1]))
                push!(G.v, parse(Int, fields[2]))
                push!(G.w, -1)
                G.m+=1
            end
            if eof(file)
                break
            end
        end
        G.n=max(maximum(G.u),maximum(G.v))
        return G
    end
end


function lapsp(G :: Graph)
	d=zeros(G.n);
	for i=1:G.m
		x=G.u[i];y=G.v[i];
		d[x]+=1;d[y]+=1;
	end
	uu=zeros(2*G.m+G.n);
	vv=zeros(2*G.m+G.n);
	ww=zeros(2*G.m+G.n);
	a=zeros(G.n);
	for i=1:G.n
		a[i]=i;
	end
	uu[1:G.m]=G.u;uu[G.m+1:2*G.m]=G.v;
	uu[2*G.m+1:2*G.m+G.n]=a;
	vv[1:G.m]=G.v;vv[G.m+1:2*G.m]=G.u;
	vv[2*G.m+1:2*G.m+G.n]=a;
	ww[1:G.m].=-G.w;ww[G.m+1:2*G.m].=-G.w;
	ww[2*G.m+1:2*G.m+G.n]=d;
    return sparse(uu,vv,ww)
end


function getS(G :: Graph)
	uu=zeros(4*G.m+2*G.n)
    vv=zeros(4*G.m+2*G.n)
    ww=zeros(4*G.m+2*G.n)
    d=zeros(G.n);
	for i=1:G.m
		x=G.u[i];y=G.v[i];
		d[x]+=1;d[y]+=1;
	end
    for i=1:G.m
        if G.w[i]>0
            uu[i]=G.u[i]
            vv[i]=G.v[i]
            ww[i]=-1
            uu[G.m+i]=G.n+G.u[i]
            vv[G.m+i]=G.n+G.v[i]
            ww[G.m+i]=-1
            uu[2*G.m+i]=G.v[i]
            vv[2*G.m+i]=G.u[i]
            ww[2*G.m+i]=-1
            uu[3*G.m+i]=G.n+G.v[i]
            vv[3*G.m+i]=G.n+G.u[i]
            ww[3*G.m+i]=-1
        elseif G.w[i]<0
            uu[i]=G.u[i]
            vv[i]=G.n+G.v[i]
            ww[i]=-1
            uu[G.m+i]=G.n+G.u[i]
            vv[G.m+i]=G.v[i]
            ww[G.m+i]=-1
            uu[2*G.m+i]=G.v[i]
            vv[2*G.m+i]=G.n+G.u[i]
            ww[2*G.m+i]=-1
            uu[3*G.m+i]=G.n+G.v[i]
            vv[3*G.m+i]=G.u[i]
            ww[3*G.m+i]=-1
        end
    end
    for i=1:G.n
        uu[4*G.m+i]=i
        vv[4*G.m+i]=i
        ww[4*G.m+i]=d[i]+1
        uu[4*G.m+G.n+i]=G.n+i
        vv[4*G.m+G.n+i]=G.n+i
        ww[4*G.m+G.n+i]=d[i]+1
    end
    return sparse(uu,vv,ww)
end


function getB_all(G)
	v1=zeros(2*G.m)
	v2=zeros(2*G.m)
	v1[1:G.m]=1:G.m;v1[G.m+1:2*G.m]=1:G.m;
	v2[1:G.m]=G.u;v2[G.m+1:2*G.m]=G.v;
	o=zeros(2*G.m)
	o[1:G.m].=1;o[G.m+1:2*G.m].=-G.w;
	return sparse(v1,v2,o)
end

function getB_pos(G)
	v1=zeros(2*G.m)
	v2=zeros(2*G.m)
	v1[1:G.m]=1:G.m;v1[G.m+1:2*G.m]=1:G.m;
	v2[1:G.m]=G.u;v2[G.m+1:2*G.m]=G.v;
	o=zeros(2*G.m)
	o[1:G.m].=1;o[G.m+1:2*G.m].=-max.(G.w,0);
	return sparse(v1,v2,o)
end

function getB_neg(G)
	v1=zeros(2*G.m)
	v2=zeros(2*G.m)
	v1[1:G.m]=1:G.m;v1[G.m+1:2*G.m]=1:G.m;
	v2[1:G.m]=G.u;v2[G.m+1:2*G.m]=G.v;
	o=zeros(2*G.m)
	o[1:G.m].=1;o[G.m+1:2*G.m].=-min.(G.w,0);
	return sparse(v1,v2,o)
end


function Uniform(n)
    x = rand(n)
    return x
end