"""Independent finite Matsubara RPA matrix check; no TPRF kernels used."""
import json
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
from vasp_slurm_agent.qe_pairing import prepare_run, KB_EV_K
root=Path(__file__).parent
nk, nw, beta, U = 4, 32, 1., 1.
coords=np.array(list(np.ndindex(nk,nk)))
energies=-2*np.cos(2*np.pi*coords/nk).sum(axis=1)
fermi=1/(1+np.exp(beta*energies))
N=nk*nk
idx=np.arange(N)
invert=np.array([np.ravel_multi_index(tuple((-c)%nk),(nk,nk)) for c in coords])
freq=(2*np.arange(-nw,nw)+1)*np.pi/beta
bosons=np.arange(-2*nw+1,2*nw)*2*np.pi/beta
chi=np.empty((len(bosons),N),complex)
for qi,q in enumerate(coords):
 shifted=np.array([np.ravel_multi_index(tuple((c+q)%nk),(nk,nk)) for c in coords])
 difference=energies[shifted]-energies
 for wi,w in enumerate(bosons):
  ratio=np.zeros(N,complex)
  den=difference-1j*w
  np.divide(fermi-fermi[shifted],den,out=ratio,where=abs(den)>1e-10)
  ratio[abs(den)<1e-10]=beta*fermi[abs(den)<1e-10]*(1-fermi[abs(den)<1e-10])
  chi[wi,qi]=ratio.mean()
assert np.max(abs(chi.imag))<1e-12
chi=chi.real
assert U*np.max(chi[2*nw-1])<1
# Standard V_s, including charge screening; TPRF uses Gamma_s = 2*V_s.
V=U+1.5*U**2*chi/(1-U*chi)-.5*U**2*chi/(1+U*chi)
qi=np.array([[np.ravel_multi_index(tuple((a-b)%nk),(nk,nk)) for b in coords] for a in coords])
widx=np.arange(2*nw)
dwi=widx[:,None]-widx[None,:]+2*nw-1
W=1/(freq[:,None]**2+energies[None,:]**2)/(beta*N)
K=-V[dwi[:,None,:,None],qi[None,:,None,:]].reshape(2*nw*N,2*nw*N)
sqrtW=np.sqrt(W.ravel());K*=sqrtW[:,None]*sqrtW[None,:]
assert np.max(abs(K-K.T))<1e-12
fi=np.arange(2*nw*N).reshape(2*nw,N)[::-1].ravel()
ki=np.arange(2*nw*N).reshape(2*nw,N)[:,invert].ravel()
# Orthogonal even-frequency/even-momentum projection on both indices.
for index in (fi,ki): K=(K+K[index,:]+K[:,index]+K[index,:][:,index])/4
vals,vectors=eigh(K,subset_by_index=[K.shape[0]-1,K.shape[0]-1])
v=vectors[:,-1]; residual=np.linalg.norm(K@v-vals[-1]*v)
report={'method':'independent direct finite Matsubara matrix','beta_per_ev':beta,'u_ev':U,'kmesh':[nk,nk,1],'matsubara_points':nw,'singlet_eigenvalue':float(vals[-1]),'eigenpair_residual':float(residual),'stoner_factor':float(U*max(chi[2*nw-1]))}
(root/'direct-reference.json').write_text(json.dumps(report,indent=2)+'\n')
p=json.loads((root/'square-32-100/request.json').read_text())['parameters']
p.update(temperature_k=1/(KB_EV_K*beta),kmesh=[nk,nk,1],matsubara_points=nw,channels=['singlet'])
prepare_run(root/'square_hr.dat',root/'square-direct-check',p,origin='model')
print(report)
