import shlex

import pytest

from vasp_slurm_agent.config import ClusterConfig
from vasp_slurm_agent.qe_workflow import _command_for, _script


@pytest.mark.parametrize("command", [
    "srun pw.x", "srun -n 8 pw.x", "srun -n8 pw.x", "srun --ntasks=8 pw.x",
    "mpirun -np 8 pw.x", "mpirun -np8 pw.x", "mpiexec -n 8 pw.x",
    "mpiexec --np=8 pw.x", "/opt/mpi/bin/mpirun -np 8 /opt/qe/bin/pw.x",
])
def test_epw_uses_one_pool_per_actual_rank(command):
    config = ClusterConfig(host="test", user="test", remote_root="/scratch/test", engine="qe", qe_pseudo_root="/pseudo", partition="cpu", qe_command=command, tasks=8)
    tokens = shlex.split(_command_for(config, "epw.x"))
    assert tokens[-2:] == ["-npool", "8"]
    assert any(token.endswith("epw.x") for token in tokens)
    assert shlex.split(_command_for(config, "ph.x")) == [
        token[:-4] + "ph.x" if token.endswith("pw.x") else token for token in shlex.split(command)
    ]


@pytest.mark.parametrize("pool", ["-nk", "-npool", "-npools"])
def test_pw_pool_layout_is_replaced_for_epw_only(pool):
    config = ClusterConfig(host="test", user="test", remote_root="/scratch/test", engine="qe", qe_pseudo_root="/pseudo", partition="cpu", qe_command=f"mpirun -np 8 pw.x {pool} 2 -ni 1 -ndiag 1", tasks=8)
    assert _command_for(config, "epw.x") == "mpirun -np 8 epw.x -ndiag 1 -npool 8"
    assert _command_for(config, "pw.x") == config.qe_command


@pytest.mark.parametrize("command,match", [
    ("mpirun -np 4 pw.x", "match"), ("srun -n 4 pw.x", "match"),
    ("pw.x", "match"), ("mpirun pw.x", "configured task"),
    ("mpiexec -n 8 -np 8 pw.x", "once"), ("mpirun -np 0 pw.x", "positive integer"),
    ("mpirun -np nope pw.x", "positive integer"), ("srun -n pw.x", "missing"),
    ("env OMP_NUM_THREADS=1 mpirun -np 8 pw.x", "direct launch"),
    ("mpirun -np 8 pw.x : -np 1 other.x", "direct launch"),
    ("mpirun -np 8 pw.x -ni 2", "one image"),
    ("mpirun -np 8 pw.x -npool", "positive integers"),
])
def test_ambiguous_or_inconsistent_epw_launch_is_rejected(command, match):
    with pytest.raises(ValueError, match=match):
        _command_for(ClusterConfig(host="test", user="test", remote_root="/scratch/test", engine="qe", qe_pseudo_root="/pseudo", partition="cpu", qe_command=command, tasks=8), "epw.x")


def test_serial_epw_and_frozen_script():
    serial = ClusterConfig(host="test", user="test", remote_root="/scratch/test", engine="qe", qe_pseudo_root="/pseudo", partition="cpu", qe_command="/opt/qe/pw.x", tasks=1)
    assert _command_for(serial, "epw.x") == "/opt/qe/epw.x -npool 1"
    parallel = ClusterConfig(host="test", user="test", remote_root="/scratch/test", engine="qe", qe_pseudo_root="/pseudo", partition="cpu", qe_command="mpirun -np 8 pw.x", tasks=8)
    script = _script(parallel, {"name": "epw_aniso", "metadata": {
        "input_file": "epw.in", "output_file": "epw.out"}})
    assert "#SBATCH --ntasks=8" in script
    assert "mpirun -np 8 epw.x -npool 8 -in epw.in > epw.out" in script


@pytest.mark.parametrize("command, expected", [
    ("srun pw.x", "srun -n 1 dos.x"),
    ("srun --mpi=pmi2 -n 32 pw.x -nk 4", "srun --mpi=pmi2 -n 1 dos.x"),
    ("srun --ntasks=32 -N 2 pw.x -ndiag 1", "srun -n 1 dos.x"),
    ("mpirun -np 8 --bind-to core /opt/qe/bin/pw.x -npool 2", "mpirun --bind-to core -n 1 /opt/qe/bin/dos.x"),
    ("mpiexec -n8 pw.x", "mpiexec -n 1 dos.x"),
    ("pw.x -nk 2", "dos.x"),
])
def test_dos_runs_on_one_rank_with_the_configured_launcher(command, expected):
    # QE 7.6 dos.x stopped (32 ranks) or hung (8 ranks) on spin-polarized PAW iron; one rank finished in a second.
    config = ClusterConfig(host="test", user="test", remote_root="/scratch/test", engine="qe", qe_pseudo_root="/pseudo", partition="cpu", qe_command=command, tasks=8)
    assert _command_for(config, "dos.x") == expected
    assert _command_for(config, "pw.x") == command
