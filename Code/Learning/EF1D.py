import numpy as np 
import matplotlib.pyplot as plt 

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

def SumCiP2(Mesh,C,x_iglob):
    S=np.zeros(len(x_iglob))
    for j in range (len(x_iglob)):
        for k in range (Mesh.NPoints//2):
            S[j]+=C[k]*P2_pair(k)(x_iglob[j])
            if (k <= Mesh.NPoinrs//2 -1):
                S[j]+=C[k]*P2_impair(k)(x_iglob[j])
            


if __name__ == "__main__" : 
    a=0
    b=1
    n=10

    Mesh=MeshP2(a,b,n)
    VisuMesh = MeshP2(a,b,1000)
    x_i=Mesh.Points
    x_iglob=VisuMesh.Points

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