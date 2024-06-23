import numpy as np
from math import *

class nn:
	def __init__(self,p,f,children):
		self.p=p
		self.f=f
		self.children=children
		self.args=[]
		self.g=[]
	
	def __call__(self,args,shift=[],counter=[0],update=False):
		self.args.clear()
		for child in self.children:
			if isinstance(child,nn):
				self.args.append(child(args),shift)
			elif isinstance(child,(int,float)):
				self.args.append(child)
		for i in range(counter[0],min(len(shift),len(self.p))):
			self.p[i]+=shift[i]
		r=self.f(self.args,self.p)
		if not update:
			for i in range(counter[0],min(len(shift),len(self.p))):
				self.p[i]-=shift[i]
		
		counter[0]+=len(self.p)
		return r
	
	def grad(self,args,gr=[],delta=1e-5,counter=[0],c=1):
		r=0
		self.args.clear()
		for child in self.children:
			if isinstance(child,nn):
				self.args.append(child(args))
			elif isinstance(child,(int,float)):
				self.args.append(child)
		
		for i in range(0,len(self.args)):
			if isinstance(child,nn):
				self.args[i]+=delta
				r=self.f(self.args,self.p)
				self.args[i]-=delta
				self.args[i]-=delta
				r-=self.f(self.args,self.p)
				self.args[i]+=delta
				r/=(delta+delta)
				r*=c
				self.children[i].grad(args,gr,delta,counter,r)
		
		for i in range(0,len(self.p)):
			self.p[i]+=delta
			r=self.f(self.args,self.p)
			self.p[i]-=delta
			self.p[i]-=delta
			r-=self.f(self.args,self.p)
			self.p[i]+=delta
			r/=(delta+delta)
			r*=c;
			if counter[0]<len(gr):
				gr[counter[0]]=r
			else:
				gr.append(r)
			counter[0]+=1


class nnet:
	def __init__(self,n):
		self.layers=[]
		k=0
		if isinstance(n,list):
			for i in range(0,len(n)):
				l=n[i]
				if isinstance(l,list) and len(l)==2 and isinstance(l[1],int) and callable(l[0]):
					if k==0:
						k=l[1]
					self.layers.append([l[0],l[1],np.zeros((l[1],k)),np.zeros(l[1]),[],np.zeros((l[1],k)),np.zeros(l[1]),np.zeros(l[1])])
					k=l[1];
	
	def __call__(self,x):
		if not len(self.layers) or len(x)!=self.layers[0][1]:
			return x
		x=np.array(x)
		for layer in self.layers:
			x=np.dot(layer[2],x)
			x+=layer[3]
			layer[4]=x
			x=np.vectorize(layer[0])(x)
			layer[7]=x
		return x
	
	def loss(self,xy,f):
		p=0
		for [x,y] in xy:
			if not len(self.layers) or len(x)!=self.layers[0][1] or len(y)!=self.layers[-1:][0][1] or not callable(f):
				continue
			p+=np.sum(np.vectorize(f)(self(x),y))
		return p
	
	def grad(self,xy,f,delta=1e-3):
		p=0
		for layer in self.layers:
			layer[5]*=0.
			layer[6]*=0.
		for [x,y] in xy:
			if not len(self.layers) or len(x)!=self.layers[0][1] or len(y)!=self.layers[-1:][0][1] or not callable(f):
				continue
			z=self(x)
			for n in range(len(self.layers)-1,-1,-1):
				layer=self.layers[n]
				if n==len(self.layers)-1:
					w=np.zeros(len(layer[4]))
					p+=np.sum(np.vectorize(f)(z,y))
					for i in range(0,len(layer[4])):
						p=z[i]
						q=(f(p+delta,y[i])-f(p-delta,y[i]))/(delta+delta)
						w[i]=q
				for i in range(0,len(layer[4])):
					p=layer[4][i]
					q=(layer[0](p+delta)-layer[0](p-delta))/(delta+delta)
					w[i]*=q
				z=self.layers[n-1][7] if 0<n else x
				for i in range(0,len(layer[4])):
					layer[6][i]+=w[i]
					for j in range(0,len(z)):
						layer[5][i][j]+=w[i]*z[j]
				if 0<n:
					w=np.dot(w,layer[2])
		return p
	def simple_descent(self,xy,f,step=1,delta=1e-3,n=8):
		self.grad(xy,f,delta)
		p,q,r=0,self.loss(xy,f),0
		for layer in self.layers:
			p+=np.linalg.norm(layer[5])**2
			p+=np.linalg.norm(layer[6])**2
		r=p
		if p>1e-16:
			p=step/sqrt(p)
			layer[5]*=p
			layer[6]*=p
		for m in range(0,n):
			for layer in self.layers:
				layer[2]-=layer[5]
				layer[3]-=layer[6]
			p=self.loss(xy,f)
			if p<q:
				q=p
				for layer in self.layers:
					layer[5]*=1.2
					layer[6]*=1.2
			else:
				for layer in self.layers:
					layer[2]+=layer[5]
					layer[5]*=-0.72
					layer[3]+=layer[6]
					layer[6]*=-0.72
		return q,r
	
# ~ ni=[]
M=20
s=lambda x:1./(exp(x)+1.)
l=lambda x,y:0.5*(x-y)**2
n=nnet([[s,M],[s,M],[s,M]])
delta=1e-2
for m in range(len(n.layers)):
	for i in range(0,M):
		n.layers[m][3][i]=np.random.uniform()
		for j in range(0,M):
			n.layers[m][2][i][j]=np.random.uniform()

print(n.layers)

xy=[]
for i in range(0,200):
	x,y=[],[]
	for j in range(0,M):
		x.append(np.random.uniform())
		y.append(np.random.uniform())
	xy.append([x,y])

n.grad(xy,l)
print(n.layers)
for m in range(0,2):
	for i in range(0,2):
		n.layers[m][3][i]+=delta
		p=n.loss(xy,l)
		n.layers[m][3][i]-=2*delta
		p-=n.loss(xy,l)
		p/=(delta+delta)
		n.layers[m][3][i]+=delta
		print(p,n.layers[m][6][i])
		for j in range(0,2):
			n.layers[m][2][i][j]+=delta
			p=n.loss(xy,l)
			n.layers[m][2][i][j]-=2*delta
			p-=n.loss(xy,l)
			p/=(delta+delta)
			n.layers[m][2][i][j]+=delta
			print(p,n.layers[m][5][i][j])

for i in range(0,200):
	print(n.simple_descent(xy,l,1./log(2+i),1e-3,16));
