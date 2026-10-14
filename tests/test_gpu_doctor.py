"""GPU checks use scheduler facts without running VASP on a login node."""

import pytest

from vasp_slurm_agent.cli import doctor
from vasp_slurm_agent.config import ClusterConfig
from vasp_slurm_agent.transport import Result


class Remote:
    def __init__(self, inventory="up|gpu:nvidia_gh200_120gb:4(S:0-3)\n", setup_ok=True):
        self.inventory = inventory
        self.setup_ok = setup_ok
        self.commands = []

    def run(self, command):
        self.commands.append(command)
        if command.startswith("sinfo "):
            return Result(0, self.inventory)
        if command.startswith("set -e") and not self.setup_ok:
            return Result(1, "", "VASP licence file is missing.")
        return Result(0, "/usr/bin/tool\n")


def cluster(**options):
    return ClusterConfig(host="cluster.example", user="researcher", remote_root="/scratch/runs",
                         vasp_command="srun --mpi=pmix vasp_std", potcar_root="/licensed/pbe",
                         partition="gpu", tasks=1, **options)


@pytest.mark.parametrize("inventory,ok", [
    ("up|gpu:nvidia_gh200_120gb:4(S:0-3)\n", True),
    ("up|gpu:1\n", True),
    ("up|gpu:a100:4,gpu:h100:4\n", True),
    ("up|(null)\n", False),
    ("down|gpu:a100:4\n", False),
    ("", False),
])
def test_gpu_partition_check(inventory, ok):
    remote = Remote(inventory)
    report = doctor(cluster(compute_backend="gpu"), remote)
    assert report["checks"]["gpu_partition"]["ok"] is ok
    assert report["ok"] is ok
    assert not any("nvidia-smi" in command for command in remote.commands)
    assert all("srun" not in command for command in remote.commands)


def test_cpu_check_does_not_require_gpu_inventory():
    remote = Remote("")
    assert doctor(cluster(), remote)["ok"]
    assert not any(command.startswith("sinfo ") for command in remote.commands)


def test_setup_failure_does_not_pass_when_executable_exists():
    remote = Remote(setup_ok=False)
    report = doctor(cluster(compute_backend="gpu", setup_commands=["test -r /private/vasp_license"]), remote)
    assert not report["ok"]
    assert not report["checks"]["vasp"]["ok"]
    assert "licence" in report["checks"]["vasp"]["detail"]
