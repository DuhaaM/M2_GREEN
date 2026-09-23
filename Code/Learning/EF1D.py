import numpy as np 
import matplotlib.pyplot as plt 
from math import sqrt
class MeshP2:  #Définition maillage 
    def __init__(self,a,b,n):
        self.a=a
        self.b=b
        self.NPoints=n+1
        self.Points=np.linspace(a,b,n+1)
        self.NElems = n//2 
        self.Elems = [] 
        for i in range(n//2):
            cell=[]
            cell.append(2*i)
            cell.append(2*i+1)
            cell.append(2*i+2)
            (self.Elems).append(cell)


def f(x):
    return np.exp(-x/10)*np.sin(np.pi*x)*np.cos(np.pi**2*x)




def P2_pair(i):
    def phi_2i(x):

        if 2*(i-1) < 0 : A=0
        else : A=2*(i-1)

        if 2*(i+1) > n : B= n
        else : B=2*(i+1)

        if  x_i[ A ] <= x <= x_i[ 2*i ] and 0 < i <= n//2 :

            return ( x - x_i[ 2*i-1 ]) * ( x - x_i[ 2*(i-1) ]) / (
                   ( x_i[ 2*i ] - x_i[ 2*i-1 ]) * ( x_i[ 2*i ] - x_i[ 2*(i-1) ]))

        if x_i[ 2*i ] <= x <= x_i[ B ] and  0 <= i <= n//2 : 

            return ( x - x_i[ 2*i+1 ]) * ( x - x_i[ 2*(i+1) ])/( 
                   ( x_i[ 2*i ] - x_i[ 2*i+1 ]) * ( x_i[ 2*i ] - x_i[ 2*(i+1) ]))        

        else:
            return 0       

    return phi_2i


def P2_impair(i):
    def phi_2i1(x):

        if 2*(i+1) > n : B= n
        else : B=2*(i+1)

        if x_i[ 2*i ] <= x < x_i[ B ] and 0 <= i <= n/2 : 

            return ( x - x_i[ 2*i ]) * ( x - x_i[ 2*(i+1) ])/( 
                   ( x_i[ 2*i+1 ] - x_i[ 2*i ]) * ( x_i[ 2*i+1 ] - x_i[ 2*(i+1) ]))        

        else:
            return 0         

    return phi_2i1

def Init(Mesh,FuncInit):
    uh = np.zeros(Mesh.NPoints)
    for i in range(Mesh.NPoints):
        uh[i] = FuncInit( Mesh.Points[i] ) 
    return uh

def SumCiP2(Mesh,C,xx):
    S=np.zeros(len(xx))
    for j in range (len(xx)):
        for k in range (Mesh.NPoints//2+1):
            S[j]+=C[2*k]*P2_pair(k)(xx[j])
            if (k <= Mesh.NPoints//2 -1):
                S[j]+=C[2*k+1]*P2_impair(k)(xx[j])
    return S


def phi(i):  # fonction de base indice i 
    if i % 2 == 0:
        return P2_pair(i//2)
    return P2_impair(i//2)

def InitA(Mesh):
    A=np.zeros((Mesh.NPoints,Mesh.NPoints))

    #Points et Poids de la méthode Gauss 3 points
    p = np.array([-np.sqrt(3/5), 0, np.sqrt(3/5)])
    w = np.array([5/9, 8/9, 5/9])

    for i in range(Mesh.NPoints):
        for j in range(Mesh.NPoints):
            if abs(j-i) > 2 :
                A[i,j]=0
            S = 0
            phii,phij=phi(i),phi(j)
            for Elem in Mesh.Elems : 
                s = 0
                if j in Elem and i in Elem :
                    xa, xb = Mesh.Points[Elem[0]], Mesh.Points[Elem[2]]
                    for k in range(3):
                        u= xa + (p[k]+1)*(xb-xa)/2
                        s+= w[k] * phii(u) * phij(u) 
                    S+= s*(xb-xa)/2
            A[i,j]= S

    return A


def Initb(f,Mesh):
    b=[]
    #Points et Poids de la méthode Gauss 3 points
    p = np.array([-np.sqrt(3/5), 0, np.sqrt(3/5)])
    w = np.array([5/9, 8/9, 5/9])
    for i in range(len(Mesh.Points)):
        S=0
        for Elem in Mesh.Elems:
            xa, xb = Mesh.Points[Elem[0]], Mesh.Points[Elem[2]]
            for k in range(3):
                u= xa + (p[k]+1)*(xb-xa)/2
                S+= (xb-xa)/2 * w[k]* phi(i)(u)*f(u)
        b.append(S)
    return b
                                    
                                
    






if __name__ == "__main__" : 
    a=0
    b=1
    n=10

    Mesh=MeshP2(a,b,n)
    VisuMesh = MeshP2(a,b,1000)
    x_i=Mesh.Points
    x_iglob=VisuMesh.Points

    print(len(Mesh.Elems[0]))

   
    print(InitA(Mesh))
    print(Initb(f,Mesh))

    print(np.linalg.solve(InitA(Mesh),Initb(f,Mesh)))

    plt.clf()
    for j in range(0,n//2+1) :     
        P2pair_i_valeurs = Init(VisuMesh,P2_pair(j))
        if(j < n//2):
             P2impair_i_valeurs= Init(VisuMesh,P2_impair(j))
        plt.plot(x_iglob,P2pair_i_valeurs)
        if(j < n//2):
            plt.plot(x_iglob,P2impair_i_valeurs)
    plt.title("Polynômes lagrangiens d'ordre 2")
    plt.savefig("P2_lagrange.png")
    plt.show()

    C=[f(x) for x in x_i]

    
    plt.clf()  
    plt.plot(x_iglob,f(x_iglob),label="Fonction à approcher ")
    plt.plot(x_iglob,SumCiP2(Mesh,C,x_iglob),label="Approximation P2")
    plt.legend()
    plt.savefig("ApproximationP2.png")
    plt.show()

    print(f"Erreur:{np.linalg.norm(f(x_iglob)-SumCiP2(Mesh,C,x_iglob),np.inf)}")
    