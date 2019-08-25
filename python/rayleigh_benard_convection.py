import numpy as np

def first_deriv(above, below, d):
    return (above - below)/(2*d)

def second_deriv(above, here, below, d):
    return (above - 2*here + below)/(d**2)

def rhs_T(psi, T, n, *args):
    dz2T = second_deriv(T[:,2:], T[:,1:-1], T[:,:-2], *args) 
    return (n*np.pi/a)*psi[:,1:-1] + dz2T - (n*np.pi/a)**2*T[:,1:-1]

def rhs_v(v, T, n, *args):
    dz2v = second_deriv(v[:,2:], v[:,1:-1], v[:,:-2], *args)
    return Ra*Pr*(n*np.pi/a)*T[:,1:-1] + Pr*(dz2v - (2*np.pi/a)**2*v[:,1:-1])

def tridiag_init(nx, nz, dz, a=2):
    sub   = np.zeros((nz-1,1))
    sup   = np.zeros((nz-1,1))
    dia   = np.zeros((nx, nz))
    work1 = np.zeros((nx, nx))
    work2 = np.zeros((nx, nx))
    sol   = np.zeros((nx, nz))


    for n in range(nx):
        dia[n,:] = (n*np.pi/a)**2 + 2/(dz**2)
    sub[:] = -1/dz**2
    sup[:] = -1/dz**2
    return sub, sup, dia, work1, work2, sol


def tridiag_solve(sub, sup, dia, work1, work2, sol, v):
    work1[:,0] = 1/dia[:,0]
    work2[:,0] = work1[:,0]*sup[0]
    for i in range(work1.shape[-1]-2):
        work1[:,i+1] = 1/(dia[:,i+1]-sub[i+1]*work2[:,i])
        work2[:,i+1] = sup[i+1]*work1[:,i+1]
    work1[:,-1] = 1/(dia[:,-1] -sub[-1]*work2[:,-2])

    sol[:,0] = v[:,0]*work1[:,0]
    for i in range(work1.shape[-1]-1):
        sol[:,i+1] = (v[:,i+1]-sub[i+1]*sol[:,i])*work1[:,i+1]
    for i in range(work1.shape[-1]-1):
        sol[:,-(1+i)] = sol[:,-(1+i)] - work2[:,-(i+1)]*sol[:,-i]
    return sol



Ra = 1e4
Pr = 1
a  = 2

nx = 50
nz = 51
dz = 1/(nz-1)

T   = np.zeros((2, nx, nz))
v   = np.zeros((2, nx, nz))
psi = np.zeros((2, nx, nz))
ns  = np.arange(nx).reshape(nx,1)

sub, sup, dia, work1, work2, sol = tridiag_init(nx, nz, dz, a=a)

dt = 1e-4

timestepping = True
while timestepping:
    T_rhs       = rhs_T(psi[1,:], T[1,:], ns, dz)
    T_rhs_last  = rhs_T(psi[0,:], T[0,:], ns, dz)
    v_rhs       = rhs_v(v[1,:], T[1,:], ns, dz)
    v_rhs_last  = rhs_v(v[0,:], T[0,:], ns, dz)

    for field in [T, v, psi]: field[0,:] = field[1,:]
    T[1,:,1:-1] = T[0,:,1:-1] + (dt/2)*(3*T_rhs - T_rhs_last)
    v[1,:,1:-1] = v[0,:,1:-1] + (dt/2)*(3*v_rhs - v_rhs_last)

    psi[1,:] = tridiag_solve(sub, sup, dia, work1, work2, sol, v[1,:])

    timestepping = False


