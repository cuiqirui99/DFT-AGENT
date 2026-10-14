from pathlib import Path
import hashlib,json,os,platform,sys,time,traceback
import numpy,scipy
from qe_pairing import solve_run
root=Path(__file__).parent
manifest=json.loads((root/'input-hashes.json').read_text())
for name,digest in manifest.items():
 actual=hashlib.sha256((root/name).read_bytes()).hexdigest()
 if actual!=digest:raise ValueError('Changed input: '+name)
environment={'python':sys.version,'numpy':numpy.__version__,'scipy':scipy.__version__,'executable_sha256':hashlib.sha256(Path(sys.executable).resolve().read_bytes()).hexdigest(),'hostname':platform.node(),'job_id':os.environ.get('SLURM_JOB_ID'),'input_hashes':manifest}
(root/'environment.json').write_text(json.dumps(environment,indent=2)+'\n')
checks=[]
for name in ['square-direct-check','square-32-100','square-32-150','square-40-150','square-4-100']:
 start=time.monotonic(); print('RUN',name,flush=True)
 try:
  result=solve_run(root/name)
  if name=='square-4-100':raise AssertionError('Unstable paramagnet was not rejected')
  checks.append({'run':name,'passed':True,'result':result,'elapsed_seconds':time.monotonic()-start})
 except ValueError as error:
  if name!='square-4-100' or 'RPA-unstable' not in str(error):raise
  checks.append({'run':name,'passed':True,'expected_rejection':str(error),'elapsed_seconds':time.monotonic()-start})
 print(json.dumps(checks[-1]),flush=True)
report={'checks':checks,'all_checks_passed':all(c['passed'] for c in checks),'model_solver_only':True,'qe_material_handoff_accepted':False,'tc_accepted':False,'release_accepted':False}
(root/'cluster-report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print('PAIRING_MODEL_CHECKS_PASSED',flush=True)
