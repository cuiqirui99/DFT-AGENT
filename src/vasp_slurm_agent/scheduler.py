"""Small Slurm/OpenPBS adapters, also copied to the remote dispatcher.

PBS syntax follows the OpenPBS qsub, qstat and qdel manuals:
https://github.com/openpbs/openpbs/tree/master/doc/man1
Scheduler status is execution evidence, never scientific acceptance.
"""
import hashlib
import json
import re
import shlex


TERMINAL_STATES = frozenset({"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
                           "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"})


class SlurmScheduler:
    name = "slurm"
    required_commands = ("sbatch", "squeue", "sacct", "scancel")
    job_id_environment = "SLURM_JOB_ID"
    tasks_environment = "SLURM_NTASKS"

    def valid_job_id(self, value):
        return isinstance(value, str) and bool(re.fullmatch(r"[0-9]+", value))

    def job_name(self, name):
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            raise ValueError("Invalid job name")
        return name

    def submit_command(self, name, script="submit.sh"):
        return ["sbatch", "--parsable", "--job-name=" + self.job_name(name), script]

    def submission_job_id(self, output):
        value = output.strip().split(";")[0]
        return value if self.valid_job_id(value) else None

    def reconcile_commands(self, name, date):
        name = self.job_name(name)
        return [["squeue", "-h", "--name", name, "-o", "%i|%j"],
                ["sacct", "-n", "-X", "-P", "--starttime", date, "--name", name,
                 "--format=JobIDRaw,JobName%100"]]

    def matching_jobs(self, output, name):
        found = set()
        for line in output.splitlines():
            fields = line.strip().split("|")
            if len(fields) >= 2 and fields[1].strip() == self.job_name(name) and self.valid_job_id(fields[0]):
                found.add(fields[0])
        return found

    def cancel_command(self, job):
        if not self.valid_job_id(job):
            raise ValueError("Invalid job identity")
        return ["scancel", job]

    def header(self, config, stdout="slurm.out", stderr="slurm.err"):
        lines = ["#!/bin/bash", "#SBATCH --partition=" + config.partition,
                 "#SBATCH --nodes=" + str(config.nodes), "#SBATCH --ntasks=" + str(config.tasks),
                 "#SBATCH --time=" + config.walltime, "#SBATCH --output=" + stdout,
                 "#SBATCH --error=" + stderr]
        if config.account:
            lines.append("#SBATCH --account=" + config.account)
        return lines + config.sbatch_resource_lines()

    def working_directory(self):
        return []


class PBSScheduler(SlurmScheduler):
    name = "pbs"
    required_commands = ("qsub", "qstat", "qdel")
    job_id_environment = "PBS_JOBID"
    tasks_environment = "PBS_NP"

    def valid_job_id(self, value):
        # Job arrays are not submitted here; never accept array wildcards.
        return isinstance(value, str) and bool(re.fullmatch(r"[0-9]+(?:\.[A-Za-z0-9][A-Za-z0-9_.-]*)?", value))

    def job_name(self, name):
        super().job_name(name)
        # Compatible with servers retaining the historical 15-character limit.
        return "dft" + hashlib.sha256(name.encode()).hexdigest()[:12]

    def submit_command(self, name, script="submit.sh"):
        return ["qsub", "-N", self.job_name(name), script]

    def submission_job_id(self, output):
        value = output.strip()
        return value if self.valid_job_id(value) else None

    def reconcile_commands(self, name, date):
        self.job_name(name)
        # Include retained history. An absent job is uncertainty, not permission
        # to resubmit after an interrupted qsub response.
        return [["qstat", "-x", "-f", "-F", "json"]]

    def jobs(self, output):
        value = json.loads(output)
        if not isinstance(value, dict) or not isinstance(value.get("Jobs", {}), dict):
            raise ValueError("Invalid PBS status response")
        jobs = value.get("Jobs", {})
        if any(not isinstance(entry, dict) for entry in jobs.values()):
            raise ValueError("Invalid PBS job attributes")
        return jobs

    def matching_jobs(self, output, name):
        expected = self.job_name(name)
        return {job for job, data in self.jobs(output).items()
                if self.valid_job_id(job) and data.get("Job_Name") == expected}

    def status_command(self, job):
        if not self.valid_job_id(job):
            raise ValueError("Invalid job identity")
        return ["qstat", "-x", "-f", "-F", "json", job]

    def parse_status(self, output, job):
        data = self.jobs(output).get(job)
        if data is None:
            return "UNKNOWN"
        state = data.get("job_state")
        if state == "F":
            # OpenPBS job.h history substates: 91 terminated, 92 finished,
            # 93 failed. A kill signal alone cannot prove user cancellation.
            if data.get("substate") in (91, "91"):
                return "CANCELLED"
            code = data.get("Exit_status")
            if type(code) is not int and not (isinstance(code, str) and re.fullmatch(r"-?[0-9]+", code)):
                return "UNKNOWN"
            return {0: "COMPLETED", -26: "OUT_OF_MEMORY", -27: "OUT_OF_MEMORY",
                    -28: "TIMEOUT", -29: "TIMEOUT"}.get(int(code), "FAILED")
        return {"Q": "PENDING", "H": "PENDING", "W": "PENDING", "T": "PENDING",
                "R": "RUNNING", "E": "COMPLETING", "S": "SUSPENDED", "U": "SUSPENDED",
                "M": "UNKNOWN"}.get(state, "UNKNOWN")

    def cancel_command(self, job):
        if not self.valid_job_id(job):
            raise ValueError("Invalid job identity")
        return ["qdel", job]

    def header(self, config, stdout="slurm.out", stderr="slurm.err"):
        per_node, remainder = divmod(config.tasks, config.nodes)
        chunks = []
        for count, ranks in ((remainder, per_node + 1), (config.nodes - remainder, per_node)):
            if count:
                chunks.append("{}:ncpus={}:mpiprocs={}:ompthreads={}".format(
                    count, ranks * config.cpus_per_task, ranks, config.cpus_per_task))
        walltime = config.walltime
        if "-" in walltime:
            days, clock = walltime.split("-")
            hours, minutes, seconds = clock.split(":")
            walltime = "{}:{}:{}".format(int(days) * 24 + int(hours), minutes, seconds)
        lines = ["#!/bin/bash", "#PBS -q " + config.partition, "#PBS -l select=" + "+".join(chunks),
                 "#PBS -l walltime=" + walltime, "#PBS -o " + stdout, "#PBS -e " + stderr]
        if config.account:
            lines.append("#PBS -A " + config.account)
        return lines + config.pbs_resource_lines()

    def working_directory(self):
        return ['cd "${PBS_O_WORKDIR:?Missing PBS working directory}"']


def get_scheduler(value="slurm"):
    """Accept a scheduler name or a ClusterConfig; reject unknown schedulers."""
    name = value if isinstance(value, str) else getattr(value, "scheduler", "slurm")
    if name == "slurm":
        return SlurmScheduler()
    if name == "pbs":
        return PBSScheduler()
    raise ValueError("Choose slurm or pbs for scheduler.")


def shell_command(argv):
    return " ".join(shlex.quote(str(value)) for value in argv)
