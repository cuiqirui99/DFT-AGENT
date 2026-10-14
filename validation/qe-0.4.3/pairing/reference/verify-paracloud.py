from pathlib import Path
import hashlib,json
import numpy as np
root=Path(__file__).parent
remote=root/'paracloud-outputs-01'
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((root/'paracloud-inputs-01/input-hashes.json').read_text())
env=json.loads((remote/'environment.json').read_text())
assert manifest==env['input_hashes']==json.loads((remote/'input-hashes.json').read_text())
assert all(digest(remote/name)==sha for name,sha in manifest.items())
assert env['job_id']=='7494956'
assert '7494956|COMPLETED|0:0|' in (root/'paracloud-sacct.txt').read_text()
report=json.loads((remote/'cluster-report.json').read_text())
assert report['all_checks_passed']
checks=[]
for entry in report['checks']:
 name=entry['run']
 if name=='square-4-100':
  assert entry['passed'] and 'RPA-unstable' in entry['expected_rejection']
  checks.append({'run':name,'rejection_verified':True}); continue
 data=json.loads((remote/name/'result.json').read_text())
 assert data==entry['result']
 local=json.loads((root/(name+'-numpy')/'result.json').read_text())
 with np.load(remote/name/'pairing.npz') as actual,np.load(root/(name+'-numpy')/'pairing.npz') as expected:
  assert np.allclose(actual['chi0'],expected['chi0'],atol=1e-13,rtol=1e-12)
  for c in data['channels']:
   same=next(x for x in local['channels'] if x['channel']==c['channel'])
   error=abs(c['eigenvalue']-same['eigenvalue'])
   assert error<1e-12
   request=json.loads((remote/name/'request.json').read_text()); mesh=tuple(request['parameters']['kmesh'])
   ag=actual['gap_'+c['channel']].ravel(); eg=expected['gap_'+c['channel']]
   if c['channel']=='triplet':
    rotation=np.array([np.ravel_multi_index(((-y)%mesh[0],x,0),mesh) for x,y,z in np.ndindex(*mesh)])
    basis=np.stack([eg.ravel(),eg[:,rotation].ravel()],axis=1)
   else: basis=eg.reshape(-1,1)
   q,_=np.linalg.qr(basis); overlap=float(np.linalg.norm(q.conj().T@ag))
   assert overlap>0.999999
   checks.append({'run':name,'channel':c['channel'],'eigenvalue':c['eigenvalue'],'local_absolute_error':error,'gap_subspace_overlap':overlap,'relative_residual':c['relative_residual']})
output={'job_id':env['job_id'],'execution_host':env['hostname'],'source_sha256':manifest['qe_pairing.py'],'frozen_inputs_verified':True,'slurm_state':'COMPLETED','slurm_exit':'0:0','checks':checks,'runtime':{k:env[k] for k in ['python','numpy','scipy','executable_sha256']},'model_solver_accepted':True,'qe_material_handoff_accepted':False,'tc_accepted':False,'release_accepted':False}
(root/'paracloud-report.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps(output,indent=2))
