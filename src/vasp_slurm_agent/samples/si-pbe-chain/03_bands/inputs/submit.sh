#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=00:30:00
#SBATCH --output=slurm.out
#SBATCH --error=slurm.err
#SBATCH --account=example-cpu
set -e
module load VASP/5.4.4-gcc-2026.03-mpich
test -s /home/researcher/dft-agent-runs/vsa-e40622e053e047b1/02_scf/CHGCAR
cp /home/researcher/dft-agent-runs/vsa-e40622e053e047b1/02_scf/CHGCAR CHGCAR
cat /home/researcher/potcar/potpaw_PBE/Si/POTCAR > POTCAR
test -s POTCAR
sha256sum POSCAR INCAR KPOINTS > input_hashes.sha256
sha256sum POTCAR > potcar_hash.sha256
grep TITEL POTCAR > potcar_titles.txt
python3 - <<'PY'
import json,os,datetime
json.dump({"job_id":os.environ.get("SLURM_JOB_ID"),"started_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),"tasks":os.environ.get("SLURM_NTASKS")},open("execution.json","w"))
PY
set +e
srun --mpi=pmi2 -m block:block vasp_std
code=$?
set -e
python3 - "$code" <<'PY'
import json,sys,datetime
p=json.load(open("execution.json"));p.update(exit_code=int(sys.argv[1]),finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
json.dump(p,open("execution.json","w"))
PY
exit "$code"
