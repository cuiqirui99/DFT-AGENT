from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ClusterConfig:
    host: str
    user: str
    remote_root: str
    vasp_command: str = ""
    potcar_root: str = ""
    partition: str = ""
    port: int = 22
    tasks: int = 8
    walltime: str = "00:30:00"
    account: str = ""
    setup_commands: list[str] = field(default_factory=list)
    connect_timeout: int = 15
    potcar_symbols: dict[str, str] = field(default_factory=dict)
    vasp_ncl_command: str = ""
    ssh_control_path: str = ""
    nodes: int = 1
    extra_sbatch: list[str] = field(default_factory=list)
    vaspkit_executable: str = "vaspkit"
    compute_backend: str = "cpu"
    gpus_per_node: int = 1
    cpus_per_task: int = 1
    engine: str = "vasp"
    qe_command: str = "srun pw.x"
    qe_bin_dir: str = ""
    qe_pseudo_root: str = ""
    qe_pseudopotentials: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.engine not in {"vasp", "qe"}:
            raise ValueError("Choose vasp or qe for engine.")
        if not isinstance(self.vaspkit_executable, str) or not self.vaspkit_executable.strip() or any(c in self.vaspkit_executable for c in "\n\r\x00"):
            raise ValueError("Enter the VASPKIT executable name or absolute path.")
        if not isinstance(self.ssh_control_path, str) or any(c in self.ssh_control_path for c in "\n\r\x00"):
            raise ValueError("Use an absolute SSH control socket path without line breaks.")
        if self.ssh_control_path:
            path = Path(self.ssh_control_path).expanduser()
            if not path.is_absolute():
                raise ValueError("Use an absolute SSH control socket path.")
            object.__setattr__(self, "ssh_control_path", str(path))
        for key in ("host", "user"):
            value = getattr(self, key)
            if not value or value.startswith("-") or re.search(r"[\s\x00-\x1f]", value):
                raise ValueError(f"Enter an SSH {key} without spaces or line breaks; it cannot start with '-'.")
        for key in ("remote_root", "potcar_root" if self.engine == "vasp" else "qe_pseudo_root"):
            value = getattr(self, key)
            if not value.startswith("/") or any(c in value for c in "\n\r\x00"):
                raise ValueError(f"Use an absolute path for {key}, starting with '/' and without line breaks.")
        for key in ("partition", "account"):
            value = getattr(self, key)
            if (key == "partition" and not value) or (value and not re.fullmatch(r"[\w.-]+", value)):
                raise ValueError(f"Enter a valid Slurm {key} using letters, numbers, dots, dashes or underscores.")
        if type(self.port) is not int or type(self.tasks) is not int or not 1 <= self.port <= 65535 or not 1 <= self.tasks <= 4096:
            raise ValueError("Use an SSH port from 1 to 65535 and an MPI task count from 1 to 4096.")
        if type(self.nodes) is not int or not 1 <= self.nodes <= self.tasks:
            raise ValueError("Nodes must be between 1 and the total MPI task count.")
        if self.compute_backend not in {"cpu", "gpu"}:
            raise ValueError("Choose cpu or gpu for compute_backend.")
        for key in ("gpus_per_node", "cpus_per_task"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"Use a positive integer for {key}.")
        if self.compute_backend == "gpu" and self.tasks != self.nodes * self.gpus_per_node:
            raise ValueError("GPU runs require one MPI task per GPU: tasks = nodes × gpus_per_node.")
        self.sbatch_resource_lines()
        if not 1 <= self.connect_timeout <= 120:
            raise ValueError("Set the SSH connection timeout to between 1 and 120 seconds.")
        if not re.fullmatch(r"(?:\d+-)?\d{1,3}:[0-5]\d:[0-5]\d", self.walltime):
            raise ValueError("Enter the time limit as HH:MM:SS or D-HH:MM:SS, for example 00:30:00.")
        if self.engine == "vasp" and (not self.vasp_command.strip() or any(c in self.vasp_command for c in "\n\r\x00")):
            raise ValueError("Enter the VASP launch command on a single line.")
        if self.engine == "qe":
            if not self.qe_command.strip() or any(c in self.qe_command for c in "\n\r\x00"):
                raise ValueError("Enter the QE launch command on a single line.")
            if self.qe_bin_dir and (not self.qe_bin_dir.startswith("/") or any(c in self.qe_bin_dir for c in "\n\r\x00")):
                raise ValueError("Use an absolute path for qe_bin_dir.")
        if not isinstance(self.qe_pseudopotentials, dict):
            raise ValueError("Map elements to QE UPF files.")
        for symbol, entry in self.qe_pseudopotentials.items():
            filename = entry.get("filename", "") if isinstance(entry, dict) else entry
            if not isinstance(symbol, str) or not re.fullmatch(r"[A-Z][a-z]?", symbol):
                raise ValueError("Use element symbols for QE pseudopotentials.")
            if not isinstance(filename, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.+-]*\.[Uu][Pp][Ff]", filename):
                raise ValueError("Use a UPF filename without a directory for each element.")
        if not isinstance(self.vasp_ncl_command, str) or any(c in self.vasp_ncl_command for c in "\n\r\x00"):
            raise ValueError("Enter vasp_ncl_command on a single line.")
        if not isinstance(self.setup_commands, list) or any(not isinstance(c, str) for c in self.setup_commands):
            raise ValueError("Provide setup_commands as a list of shell commands for your cluster environment.")
        if any(line.lstrip().startswith("#SBATCH") for command in self.setup_commands for line in command.splitlines()):
            raise ValueError("Put #SBATCH options in extra_sbatch, not environment setup commands.")
        for symbol, label in self.potcar_symbols.items():
            if not re.fullmatch(r"[A-Z][a-z]?", symbol) or not re.fullmatch(r"[A-Za-z0-9_.+-]+", label):
                raise ValueError('Map each element to a POTCAR folder name, for example {"Ti": "Ti_pv"}.')

    def sbatch_resource_lines(self) -> list[str]:
        if not isinstance(self.extra_sbatch, list):
            raise ValueError("Provide extra_sbatch as a list of Slurm options.")
        allowed = {"mem", "mem-per-cpu", "mem-per-gpu", "qos", "constraint", "gpus",
                   "gpus-per-node", "gpus-per-task", "gres", "ntasks-per-node", "cpus-per-task",
                   "exclusive", "hint", "reservation", "licenses", "distribution"}
        lines = []
        seen = set()
        for line in self.extra_sbatch:
            if not isinstance(line, str) or re.search(r"[\x00-\x1f\x7f]", line):
                raise ValueError("Use one Slurm option per line.")
            option = line.strip().removeprefix("#SBATCH").strip()
            match = re.fullmatch(r"--([a-z][a-z-]*)(?:[= ]+([^\s#]+))?", option)
            if match and match[1] == "cpu-bind":
                raise ValueError("Use --cpu-bind with srun in the VASP command, not in extra_sbatch.")
            if not match or match[1] not in allowed or match[1] in seen:
                raise ValueError("Use a supported resource option once; set nodes, tasks, partition, account and time in their own fields.")
            key, value = match.groups()
            managed = {"gpus", "gpus-per-node", "gpus-per-task", "gres", "ntasks-per-node", "cpus-per-task"}
            if self.compute_backend == "gpu" and key in managed:
                raise ValueError(f"Set --{key} through the GPU resource fields, not extra_sbatch.")
            if self.compute_backend == "cpu" and self.cpus_per_task != 1 and key == "cpus-per-task":
                raise ValueError("Set CPUs per task once, in cpus_per_task or extra_sbatch.")
            if value is None and key != "exclusive":
                raise ValueError(f"Give --{key} a value.")
            if key in {"ntasks-per-node", "cpus-per-task"}:
                if not value.isdigit() or int(value) < 1:
                    raise ValueError(f"Use a positive integer for --{key}.")
                if key == "ntasks-per-node" and int(value) * self.nodes < self.tasks:
                    raise ValueError("Nodes × tasks per node must accommodate the total MPI tasks.")
            seen.add(key)
            lines.append(f"#SBATCH --{key}" + (f"={value}" if value is not None else ""))
        if len(seen & {"mem", "mem-per-cpu", "mem-per-gpu"}) > 1:
            raise ValueError("Choose one Slurm memory option.")
        if {"gres", "gpus-per-node"}.issubset(seen):
            raise ValueError("Choose --gres or --gpus-per-node, not both.")
        if self.compute_backend == "gpu":
            # A node request reserves all four GPUs on Arrhenius. Let Slurm
            # place a single-node request from the GPU and task counts instead.
            if self.nodes == 1:
                lines.append(f"#SBATCH --gpus={self.gpus_per_node}")
            else:
                lines.extend([f"#SBATCH --gpus-per-node={self.gpus_per_node}",
                              f"#SBATCH --ntasks-per-node={self.gpus_per_node}"])
            lines.append(f"#SBATCH --cpus-per-task={self.cpus_per_task}")
        elif self.cpus_per_task != 1:
            lines.append(f"#SBATCH --cpus-per-task={self.cpus_per_task}")
        return lines

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def load(cls, path: str | Path) -> "ClusterConfig":
        return cls(**json.loads(Path(path).expanduser().read_text()))

    def save(self, path: str | Path) -> Path:
        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        target.chmod(0o600)
        return target
