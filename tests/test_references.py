"""Official QE, Elk, ABINIT and CP2K references: parsing, pinned builds, local search and the planners' use of them.

Every manual, example and keyword text here is synthetic and written for these tests; the real texts
are GPL and are only ever downloaded or read on the user's computer.
"""
import gzip
import hashlib
import io
import json
import re
import shutil
import tarfile
from types import SimpleNamespace

import pytest

from vasp_slurm_agent import agent, engine_agent, references
from vasp_slurm_agent.transport import Result

MANUAL = r"""! Copyright line of a test file.
program elk
end program
!BOI
!   \section{Units}
!   Unless stated otherwise, lengths are in Bohr and energies in Hartree. One
!   Hartree is $27.2$~eV and ${\bf k}$ vectors are in $\AA^{-1}$ only in plots.
!
!   \section{Input blocks}
!   Each block has a name and values.
!
!   \block{lradstp}{
!   {\tt lradstp} & radial step length for the coarse mesh & integer & 4}
!   Functions are computed on a coarse radial mesh and interpolated. Too large
!   a step loses precision; see {\tt nrmtscf} and $\alpha\le\frac{1}{2}$.
!
!   \block{tasks}{
!   {\tt task(i) } & the $i$th task & integer & $-1$}
!   A list of tasks:
!   \vskip 6pt
!   \begin{tabularx}{\textwidth}[h]{lX}
!   0 & Ground state from atomic densities. \\
!   31/2/3 & 1/2/3D density plot. \\
!   460/1/2/3 & Time evolution under a laser
!    pulse. \\
!   \end{tabularx}
!
!   \block{wplot}{
!   {\tt nwplot} & number of points & integer & 500 \\
!   {\tt ngrkf} & fine grid size & integer & 100 \\
!   \hline
!   {\tt wplot} & energy window for plots & real(2) & $(-0.5,0.5)$}
!   Window and resolution of DOS and optics plots.
!
!   \section{Contributing to Elk}
!   Not part of the reference.
!EOI
"""

EXAMPLE = """
! Optical spectrum of lithium fluoride with excitons from the bootstrap
! kernel.

tasks
  0         ! ground state
  120
  121
  320

fxctype
  210

lradstp
  2

atoms
  2
  'Li.in'
  1
  0.0 0.0 0.0
  'F.in'
  1
  0.5 0.5 0.5
"""

BSE_EXAMPLE = """! Bethe-Salpeter optical spectrum of lithium fluoride with excitons.

tasks
  0
  185
  186

atoms
  2
  'Li.in'
  1
  0.0 0.0 0.0
  'F.in'
  1
  0.5 0.5 0.5
"""

XML = b"""<?xml version="1.0" encoding="utf-8"?>
<CP2K_INPUT>
 <CP2K_VERSION>CP2K version 2026.2</CP2K_VERSION>
 <REFERENCE key="Test2026"><TITLE>A test</TITLE></REFERENCE>
 <SECTION repeats="no">
  <NAME>FORCE_EVAL</NAME>
  <DESCRIPTION>Parameters of the forces.</DESCRIPTION>
  <LOCATION>input_cp2k.F:1</LOCATION>
  <SECTION repeats="no">
   <NAME>SCF</NAME>
   <DESCRIPTION>Self-consistent field settings. See [OT](#CP2K_INPUT.FORCE_EVAL.SCF.OT).</DESCRIPTION>
   <LOCATION>input_cp2k_scf.F:10</LOCATION>
   <KEYWORD repeats="no" removed="no">
    <NAME type="default">EPS_SCF</NAME>
    <DATA_TYPE kind="real"><N_VAR>1</N_VAR></DATA_TYPE>
    <USAGE>EPS_SCF 1.e-6</USAGE>
    <DESCRIPTION>Target accuracy of the SCF cycle.</DESCRIPTION>
    <DEFAULT_VALUE>1.0E-005</DEFAULT_VALUE>
    <LOCATION>input_cp2k_scf.F:20</LOCATION>
   </KEYWORD>
   <SECTION repeats="no">
    <NAME>OT</NAME>
    <DESCRIPTION>Orbital transformation for insulators.</DESCRIPTION>
    <LOCATION>input_cp2k_scf.F:30</LOCATION>
    <SECTION_PARAMETERS repeats="no" removed="no">
     <NAME type="default">SECTION_PARAMETERS</NAME>
     <DATA_TYPE kind="logical"><N_VAR>1</N_VAR></DATA_TYPE>
     <DESCRIPTION>Switches OT on.</DESCRIPTION>
     <DEFAULT_VALUE>F</DEFAULT_VALUE>
     <LONE_KEYWORD_VALUE>T</LONE_KEYWORD_VALUE>
    </SECTION_PARAMETERS>
    <KEYWORD repeats="no" removed="no">
     <NAME type="default">PRECONDITIONER</NAME>
     <NAME type="alias">PRECOND</NAME>
     <DATA_TYPE kind="keyword"><ENUMERATION strict="yes">
      <ITEM><NAME>FULL_ALL</NAME><DESCRIPTION>Most effective preconditioner.</DESCRIPTION></ITEM>
      <ITEM><NAME>FULL_KINETIC</NAME><DESCRIPTION>Cheap and robust.</DESCRIPTION></ITEM>
     </ENUMERATION><N_VAR>1</N_VAR></DATA_TYPE>
     <USAGE>PRECONDITIONER FULL_ALL</USAGE>
     <DESCRIPTION>Preconditioner of the minimization.</DESCRIPTION>
     <DEFAULT_VALUE>FULL_KINETIC</DEFAULT_VALUE>
     <LOCATION>input_cp2k_scf.F:40</LOCATION>
    </KEYWORD>
   </SECTION>
  </SECTION>
 </SECTION>
 <SECTION repeats="no">
  <NAME>ATOM</NAME>
  <DESCRIPTION>The atomic code.</DESCRIPTION>
  <LOCATION>input_cp2k_atom.F:1</LOCATION>
  <SECTION repeats="no">
   <NAME>SCF</NAME>
   <DESCRIPTION>Self-consistent field settings. See [OT](#CP2K_INPUT.FORCE_EVAL.SCF.OT).</DESCRIPTION>
   <LOCATION>input_cp2k_scf.F:10</LOCATION>
   <KEYWORD repeats="no" removed="no">
    <NAME type="default">EPS_SCF</NAME>
    <DATA_TYPE kind="real"><N_VAR>1</N_VAR></DATA_TYPE>
    <USAGE>EPS_SCF 1.e-6</USAGE>
    <DESCRIPTION>Target accuracy of the SCF cycle.</DESCRIPTION>
    <DEFAULT_VALUE>1.0E-005</DEFAULT_VALUE>
    <LOCATION>input_cp2k_scf.F:20</LOCATION>
   </KEYWORD>
  </SECTION>
 </SECTION>
 <SECTION repeats="no">
  <NAME>MOTION</NAME>
  <DESCRIPTION>Atomic motion.</DESCRIPTION>
  <LOCATION>input_cp2k_motion.F:1</LOCATION>
  <SECTION repeats="no">
   <NAME>MD</NAME>
   <DESCRIPTION>Molecular dynamics.</DESCRIPTION>
   <LOCATION>input_cp2k_md.F:1</LOCATION>
   <KEYWORD repeats="no" removed="no">
    <NAME type="default">TIMESTEP</NAME>
    <DATA_TYPE kind="real"><N_VAR>1</N_VAR></DATA_TYPE>
    <DESCRIPTION>Time step of the integration.</DESCRIPTION>
    <DEFAULT_VALUE>5.0E-001</DEFAULT_VALUE>
    <DEFAULT_UNIT>fs</DEFAULT_UNIT>
    <LOCATION>input_cp2k_md.F:9</LOCATION>
   </KEYWORD>
   <SECTION repeats="no">
    <NAME>PRINT</NAME>
    <DESCRIPTION>Output of molecular dynamics.</DESCRIPTION>
    <LOCATION>input_cp2k_md.F:50</LOCATION>
    <KEYWORD repeats="no" removed="no">
     <NAME type="default">TIMESTEP</NAME>
     <DATA_TYPE kind="integer"><N_VAR>1</N_VAR></DATA_TYPE>
     <DESCRIPTION>Write every this many time steps.</DESCRIPTION>
     <DEFAULT_VALUE>1</DEFAULT_VALUE>
     <LOCATION>input_cp2k_md.F:60</LOCATION>
    </KEYWORD>
   </SECTION>
  </SECTION>
 </SECTION>
</CP2K_INPUT>
"""

CP2K_INPUT = """# A small water box under a thermostat.
@SET cut 400
&GLOBAL
  PROJECT water
  RUN_TYPE MD
&END GLOBAL
&MOTION
  &MD
    ENSEMBLE NVT
    TIMESTEP 0.5
    TEMPERATURE 300
    &THERMOSTAT
      TYPE CSVR
    &END THERMOSTAT
  &END MD
&END MOTION
&FORCE_EVAL
  METHOD QS
  &DFT
    &MGRID
      CUTOFF ${cut}
    &END MGRID
    &SCF
      &OT ON
      &END OT
    &END SCF
    &XC
      &XC_FUNCTIONAL PBE
      &END XC_FUNCTIONAL
    &END XC
  &END DFT
  &SUBSYS
    &COORD
""" + "".join(f"      {'O' if index % 3 == 0 else 'H'} {index}.0 0.0 0.0\n" for index in range(9)) + """    &END COORD
    &KIND H
    &END KIND
    &KIND O
    &END KIND
  &END SUBSYS
&END FORCE_EVAL
"""

MARKDOWN = """# Molecular Dynamics

Molecular dynamics moves atoms in time.

```{youtube} abcdef
---
align: center
---
```

## Thermostats

A weak [CSVR](#CP2K_INPUT.MOTION.MD.THERMOSTAT.CSVR) coupling keeps the temperature.

```none
## not a heading inside code
&MD
  ENSEMBLE NVT
&END MD
```

## Equilibration

Discard the first picoseconds before averaging.
"""

# ABINIT keeps its variables as Python code; DFT Agent reads it as data and never runs it.
VARIABLES = r"""# coding: utf-8
import os
os.system("touch should-never-exist")
from abimkdocs.variables import ValueWithUnit, MultipleValue, Range, ValueWithConditions
Variable = dict
variables = [
Variable(
    abivarname="tsmear",
    varset="gstate",
    vartype="real",
    topics=['BandOcc_basic'],
    dimensions="scalar",
    defaultval=0.01,
    mnemonics="Temperature of SMEARing",
    characteristics=['[[ENERGY]]'],
    added_in_version="before_v9",
    text=r'''
Broadening of the occupations of metals, for [[occopt]] 3 to 7. See [[cite:Test2020]] and [[tutorial:base4|the
tutorial on metals]]; set it with [[occopt].
''',
),
Variable(
    abivarname="ngkpt",
    varset="basic",
    vartype="integer",
    topics=['k-points_basic'],
    dimensions=[3],
    defaultval=MultipleValue(number=3, value=0),
    mnemonics="Number of Grid points for K PoinTs",
    requires="[[kptopt]] >= 0",
    excludes="[[kptrlatt]]",
    added_in_version="before_v9",
    text=r'''Divisions of the k-point grid along the reciprocal axes.''',
),
Variable(
    abivarname="nspinor",
    varset="gstate",
    vartype="integer",
    topics=['spinpolarisation_basic'],
    dimensions="scalar",
    defaultval=ValueWithConditions({'[[pawspnorb]] == 1': 2, 'defaultval': 1}),
    mnemonics="Number of SPINORial components",
    added_in_version="before_v9",
    text=r'''Two spinor components for spin-orbit coupling, with [[so_psp]].''',
),
Variable(
    abivarname="ecutsm",
    varset="dev",
    vartype="real",
    topics=['Planewaves_expert'],
    dimensions="scalar",
    defaultval=ValueWithUnit(units='Ha', value=0.5),
    mnemonics="ECUT SMearing",
    characteristics=['[[DEVELOP]]', '[[ENERGY]]'],
    added_in_version="before_v9",
    text=r'''Smearing of the kinetic energy for cell optimizations, in steps of [[ecut]].''',
),
Variable(
    abivarname="gwgrid",
    varset="gw",
    vartype="integer",
    topics=['GW_useful'],
    dimensions=[Range({'start': 1, 'stop': '[[nkpt]]'})],
    defaultval=__import__("os").getcwd(),
    mnemonics="GW GRID",
    added_in_version="10.0",
    text=r'''A test variable whose default is code.''',
),
]
"""

ANADDB_VARIABLES = r"""executable = "anaddb"
Variable = dict
variables = [
Variable(
    abivarname="asr@anaddb",
    varset="anaddb",
    vartype="integer",
    topics=['Phonons_basic', 'PhononBands_basic'],
    dimensions="scalar",
    defaultval=1,
    mnemonics="Acoustic Sum Rule",
    added_in_version="before_v9",
    text=r'''Imposes the acoustic sum rule on the force constants; see also [[asr]] for abinit.''',
),
]
"""

TUTORIAL = """---
authors: Test
---

# Test tutorial on phonons:

## Phonons of a III-V semiconductor.

This tutorial computes phonons of AlAs with density-functional perturbation theory, then [[anaddb:ifcflag]].

[TUTORIAL_README]

!!! important

    Converge [[ecut]] first, see [[topic:Phonons]].

{% dialog tests/tutorespfn/Input/tphon_1.abi tests/tutorespfn/Input/tphon_2.abi %}

## 1 Interatomic force constants

Fourier interpolation of the dynamical matrices <!-- a hidden remark --> gives the phonon bands.
"""

TOPIC = """---
description: How to compute phonon frequencies with this test package
authors: Test
---

This page gives hints on phonons.

## Introduction

Phonons come from second derivatives of the energy with respect to atomic displacements.

## Related Input Variables

{{ related_variables }}

## Selected Input Files

{{ selected_input_files }}
"""

PHONON_INPUT = """# Crystalline AlAs: phonons at all q points of a 4x4x4 mesh
# and the electric-field response.

  ndtset 2
  rfphon2 1           # phonons
  rfelfd2 3           # electric field
  kptopt2 3
  nqpt 1  ngqpt 4 4 4
  ecut 3.0  tsmear 0.01
  ngkpt 4 4 4
  natom 2  ntypat 2  znucl 13 33  typat 1 2
  pseudos "Al.psp8, As.psp8"

##############################################################
# This section is used only for regression testing of ABINIT #
##############################################################
#%%<BEGIN TEST_INFO>
#%% [setup]
#%% executable = abinit
#%% test_chain = tphon_1.abi, tphon_2.abi
#%% [extra_info]
#%% keywords = NC, DFPT
#%% description = Crystalline AlAs: phonons
#%%<END TEST_INFO>
"""

ANADDB_INPUT = """! Interpolation of the AlAs phonon bands
 ifcflag 1    ! interatomic force constants
 ngqpt 4 4 4
 asr 1
 nqpath 3
 qpath 0 0 0  0.5 0 0  0.5 0.5 0

#%%<BEGIN TEST_INFO>
#%% [setup]
#%% executable = anaddb
#%% test_chain = tphon_1.abi, tphon_2.abi
#%% [extra_info]
#%% description = AlAs phonon bands
#%%<END TEST_INFO>
"""

PAW_INPUT = """# Aluminium arsenide with PAW and GW
 optdriver 4
 ecut 8 pawecutdg 16
 natom 2 ntypat 2 znucl 13 33
 pseudos "Al.xml, As.xml"
"""

NAMES = {"abinit": frozenset("ndtset rfphon rfelfd kptopt nqpt ngqpt ecut tsmear ngkpt natom ntypat znucl typat pseudos "
                             "optdriver pawecutdg".split()),
         "anaddb": frozenset("ifcflag ngqpt asr nqpath qpath".split())}


def _more(source, program):
    """The variables file with plain entries for the other names the test inputs use."""
    known = set(re.findall(r'abivarname="([a-z_0-9]+)', source))
    extra = "".join(f'Variable(abivarname="{name}{"" if program == "abinit" else "@" + program}", varset="basic", vartype="integer", '
                    f'topics={["Phonons_useful"] if program == "anaddb" else []}, dimensions="scalar", mnemonics="{name}", '
                    f'added_in_version="1", text="Test."),\n'
                    for name in sorted(NAMES[program] - known))
    return source[:source.rindex("]")] + extra + "]\n"


ABINIT_FILES = {"abimkdocs/variables_abinit.py": _more(VARIABLES, "abinit"),
                "abimkdocs/variables_anaddb.py": _more(ANADDB_VARIABLES, "anaddb"),
                "doc/tutorial/tphon.md": TUTORIAL, "doc/topics/_Phonons.md": TOPIC,
                "tests/tutorespfn/Input/tphon_1.abi": PHONON_INPUT, "tests/tutorespfn/Input/tphon_2.abi": ANADDB_INPUT,
                "tests/tutorial/Input/tgw_1.abi": PAW_INPUT}


# QE 7.4.1's INPUT_PW.def opens with "&input_description"; helpdoc reads it like the plain word.
HELPDOC = r"""&input_description -distribution {Test} -package PWscf -program pw.x {
    toc {}
    intro {
        All quantities are in @b {Rydberg atomic units} unless stated.
    }
    namelist SYSTEM {
        var ecutwfc -type REAL {
            dimensionality { energy }
            units { Ry }
            status { REQUIRED }
            info {
                kinetic energy cutoff for wavefunctions, see @ref ecutrho
            }
        }
        var occupations -type CHARACTER {
            options {
                info { Available options are: }
                opt -val 'smearing' { gaussian smearing for metals; see @ref degauss }
                opt -val 'fixed' { for insulators }
            }
        }
        var degauss -type REAL {
            units { Ry }
            default { 0.D0 }
            info { value of the gaussian spreading in metals }
        }
        dimension starting_magnetization -start 1 -end ntyp -type REAL {
            default { 0 }
            info { Starting spin polarization on atomic type 'i'. }
        }
    }
    namelist ELECTRONS {
        var mixing_beta -type REAL {
            default { 0.7D0 }
            info { mixing factor for self-consistency }
        }
    }
    card K_POINTS {
        flag kpoint_type -use optional {
            enum { tpiba | automatic | gamma }
            default { tpiba }
            options {
                info { K_POINTS options are: }
                opt -val automatic { automatically generated uniform grid }
                opt -val gamma { use k = 0 }
            }
        }
        choose {
            elsewhen -test "automatic" {
                syntax -flag {automatic} {
                    line {
                        vargroup -type INTEGER {
                            var nk1 { }
                            var nk2 { }
                            var nk3 { }
                            info { The k-point grid as in Monkhorst-Pack grids. }
                        }
                    }
                }
            }
        }
    }
}
"""

DOS_HELPDOC = r"""input_description -distribution {Test} -package PP -program dos.x {
    namelist DOS {
        var degauss -type REAL {
            units { Ry }
            default { 0.0 }
            info { gaussian broadening, Ry (not eV!) }
        }
    }
}
"""

README = """This example shows how to calculate phonon dispersions of AlAs.

The calculation proceeds as follows:

1) make a self-consistent calculation (input=alas.scf.in, output=alas.scf.out)

2) make a phonon calculation on a 4x4x4 grid of q-points (input=alas.ph.in)
"""

RUN_EXAMPLE = """#!/bin/sh
cat > alas.scf.in << EOF
 &control
    calculation='scf'
    pseudo_dir = '$PSEUDO_DIR/',
 /
 &system
    ibrav=  2, celldm(1) =10.50, nat=  2, ntyp= 2,
    ecutwfc =16.0
 /
 &electrons
    conv_thr =  1.0d-8
 /
ATOMIC_SPECIES
 Al  26.98  Al.pz-vbc.UPF
 As  74.92  As.pz-bhs.UPF
ATOMIC_POSITIONS alat
 Al 0.00 0.00 0.00
 As 0.25 0.25 0.25
K_POINTS automatic
 4 4 4 1 1 1
EOF
$ECHO "  running the scf calculation...\\c"
cat > alas.ph.in << EOF
phonons of AlAs
 &inputph
  tr2_ph=1.0d-12,
  ldisp=.true.,
  nq1=4, nq2=4, nq3=4
 /
EOF
$ECHO "  running the phonon calculation ...\\c"
cat > si.md.in << EOF
 &control
    calculation='md'
 /
 &system
    ibrav=2, celldm(1)=10.2, nat=2, ntyp=1, ecutwfc=12.0
 /
 &electrons
 /
 &ions
 /
ATOMIC_SPECIES
 Si  28.086  Si.pz-vbc.UPF
ATOMIC_POSITIONS alat
 Si 0.00 0.00 0.00
 Si 0.25 0.25 0.25
K_POINTS gamma
EOF
$ECHO "  running molecular dynamics of Si...\\c"
cat > plot.gnu << EOF
plot 'dispersion' with lines
EOF
"""

EPW_SCF = """ &control
    calculation='scf', prefix='diam'
 /
 &system
    ibrav=2, celldm(1)=6.64, nat=2, ntyp=1, ecutwfc=40
 /
 &electrons
 /
ATOMIC_SPECIES
  C 12.01 C.UPF
ATOMIC_POSITIONS alat
  C 0.00 0.00 0.00
  C 0.25 0.25 0.25
K_POINTS automatic
 6 6 6 0 0 0
"""

EPW_INPUT = """--
 &inputepw
  prefix = 'diam'
  elph = .true.
  wannierize = .true.
  eliashberg = .true.
  muc = 0.1
 /
"""

QE_FILES = {"PW/Doc/INPUT_PW.def": HELPDOC, "PP/Doc/INPUT_DOS.def": DOS_HELPDOC, "PHonon/examples/example02/README": README,
            "PHonon/examples/example02/run_example": RUN_EXAMPLE, "EPW/examples/test/phonons/scf.in": EPW_SCF,
            "EPW/examples/test/epw/epw.in": EPW_INPUT}


def _tar(path, root, members):
    with tarfile.open(path, "w:gz") as tar:
        for name, text in members.items():
            data = text.encode()
            info = tarfile.TarInfo(f"{root}/{name}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    data = path.read_bytes()
    return {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}


@pytest.fixture
def pinned(tmp_path, monkeypatch):
    """A catalog whose Elk release and CP2K files are already in the download cache."""
    cache = tmp_path / "cache"
    (cache / "elk").mkdir(parents=True)
    archive = _tar(cache / "elk" / "elk-9.9.9.tgz", "elk-9.9.9", {
        "src/elk.f90": MANUAL, "examples/TDDFT-optics/LiF-bootstrap/elk.in": EXAMPLE,
        "examples/BSE/LiF/elk.in": BSE_EXAMPLE, "examples/TDDFT-optics/LiF-bootstrap/EPSILON.OUT": "1 2\n"})
    tag = "v2026.2"
    (cache / "cp2k-docs" / tag).mkdir(parents=True)
    documents = []
    for path, text in (("docs/methods/sampling/molecular_dynamics.md", MARKDOWN), ("benchmarks/QS/water.inp", CP2K_INPUT)):
        data = text.encode()
        (cache / "cp2k-docs" / tag / path.replace("/", "__")).write_bytes(data)
        documents.append({"path": path, "sha256": hashlib.sha256(data).hexdigest(), "size": len(data)})
    pins = {}
    for engine, version, sources in (("abinit", "9.9.9", ABINIT_FILES), ("qe", "9.9", QE_FILES)):
        (cache / f"{engine}-docs" / version).mkdir(parents=True)
        pins[engine] = []
        for path, text in sources.items():
            data = text.encode()
            (cache / f"{engine}-docs" / version / path.replace("/", "__")).write_bytes(data)
            pins[engine].append({"path": path, "sha256": hashlib.sha256(data).hexdigest(), "size": len(data)})
    catalog = {
        "schema_version": 1,
        "qe": {"license": {"name": "GPL-2.0-or-later", "url": "https://qe.invalid/"}, "manual": "https://qe.invalid/Doc/",
               "repository": "https://github.invalid/QEF/q-e", "raw": "https://raw.invalid/qe/{commit}/", "default": "9.9",
               "releases": {"9.9": {"commit": "beef", "files": pins["qe"]}},
               "lessons": [{"id": "qe-hybrid", "title": "Hybrid exchange thresholds", "text": "HSE06 exchange levelled off near 1e-7 Ry.",
                            "tags": ["hybrid", "hse06"], "elements": ["Ga", "As"]}]},
        "elk": {"license": {"name": "GPL-3.0-or-later", "url": "https://elk.invalid/"}, "manual": "https://elk.invalid/",
                "default": "9.9.9", "releases": {"9.9.9": {"filename": "elk-9.9.9.tgz", "url": "https://elk.invalid/elk-9.9.9.tgz",
                                                           "root": "elk-9.9.9", **archive}},
                "lessons": [{"id": "elk-lithium", "title": "Lithium needs lradstp 2", "text": "LiF did not converge with the default mesh.",
                             "tags": ["lradstp", "lithium"], "elements": ["Li"], "evidence": "docs/validation-0.4.5.md"}]},
        "abinit": {"license": {"name": "GPL-3.0-only", "url": "https://abinit.invalid/"}, "manual": "https://docs.abinit.invalid/",
                   "repository": "https://github.invalid/abinit/abinit", "raw": "https://raw.invalid/abinit/{commit}/",
                   "default": "9.9.9", "releases": {"9.9.9": {"commit": "c0ffee", "files": pins["abinit"]}},
                   "lessons": [{"id": "abinit-metal-phonons", "title": "Metal phonons need a dense k mesh",
                                "text": "Aluminium phonons needed a 16x16x16 k mesh and broadening.",
                                "tags": ["phonons", "metal"], "elements": ["Al"]}]},
        "cp2k": {"license": {"name": "GPL-2.0-or-later", "url": "https://cp2k.invalid/"}, "manual": "https://manual.cp2k.invalid/",
                 "documents": {"tag": tag, "base": "https://raw.invalid/", "files": documents},
                 "lessons": [{"id": "cp2k-water", "title": "Equilibrate liquid water first",
                              "text": "A packed water box needs a strongly thermostatted MD stage before production.",
                              "tags": ["md", "water", "equilibration"], "elements": ["H", "O"]}]},
    }
    monkeypatch.setattr(references, "catalog", lambda: catalog)
    monkeypatch.setattr(references, "_REFERENCES", {})
    monkeypatch.setattr(references, "_LOADED", {})

    def offline(*args, **kwargs):
        raise AssertionError("Pinned files must come from the cache in tests.")
    monkeypatch.setattr("vasp_slurm_agent.engine_setup.urllib.request.urlopen", offline)
    return SimpleNamespace(cache=cache, catalog=catalog)


def test_elk_manual_blocks_tasks_and_sections():
    records = {record["id"]: record for record in references.parse_elk_manual(MANUAL)}
    block = records["keyword:lradstp"]
    assert block["title"] == "lradstp: radial step length for the coarse mesh"
    assert (block["type"], block["default"]) == ("integer", "4")
    assert "see nrmtscf and α≤(1)/(2)" in block["text"] and "{" not in block["text"] and "\\" not in block["text"]
    assert block["source"].startswith("src/elk.f90:")
    assert records["keyword:wplot"]["aliases"] == ["ngrkf", "nwplot"]
    assert [field["name"] for field in records["keyword:wplot"]["fields"]] == ["nwplot", "ngrkf", "wplot"]
    assert records["keyword:tasks"]["title"] == "tasks"
    # Task rows become their own entries; "31/2/3" names tasks 31, 32 and 33.
    assert records["task:31/2/3"]["numbers"] == [31, 32, 33]
    assert records["task:460/1/2/3"]["numbers"] == [460, 461, 462, 463]
    assert records["task:460/1/2/3"]["text"] == "Time evolution under a laser pulse."
    assert "0: Ground state from atomic densities." in records["keyword:tasks"]["text"]
    assert records["guide:units"]["text"].startswith("Unless stated otherwise") and "27.2 eV" in records["guide:units"]["text"]
    assert not any("Contributing" in record["title"] for record in records.values())


def test_elk_example_description_tasks_and_species():
    record = references.parse_elk_example("examples/TDDFT-optics/LiF-bootstrap/elk.in", EXAMPLE)
    assert record["id"] == "example:TDDFT-optics/LiF-bootstrap" and record["tags"] == ["TDDFT-optics"]
    assert record["summary"].startswith("Optical spectrum of lithium fluoride with excitons from the bootstrap")
    assert record["tasks"] == [0, 120, 121, 320] and record["elements"] == ["F", "Li"]
    assert record["blocks"] == ["tasks", "fxctype", "lradstp", "atoms"]


def test_elk_versions_come_from_modules_and_managed_folders():
    module = SimpleNamespace(setup_commands=["module load Elk/11.0.2-gcc-2026.03-mpich-eb"], engine_command="srun elk")
    managed = SimpleNamespace(setup_commands=["unset LD_LIBRARY_PATH"], engine_command="/work/software/elk-10.2.4-ab12cd34/bin/elk")
    plain = SimpleNamespace(setup_commands=[], engine_command="srun elk", host="h", user="u")
    assert references.environment_version("elk", module) == "11.0.2"
    assert references.environment_version("elk", managed) == "10.2.4"
    assert references.environment_version("cp2k", module) is None and references.environment_version("elk", plain) is None
    assert references.release("elk", "11.0.2") == "11.0.2"
    assert references.release("elk", "10.8.16") == "10.2.4"   # the newest pinned manual not newer than the program
    assert references.release("elk", "12.1") == "11.0.2" and references.release("elk", "9.1") == "10.2.4"
    assert references.release("elk", None) == references.catalog()["elk"]["default"]
    site = SimpleNamespace(setup_commands=["module load ABINIT/10.6.7-gcc-2026.03-mpich-eb"], engine_command="srun abinit")
    folder = "/work/software/abinit-10.0.3-2e0b2060/bin/"
    installed = SimpleNamespace(setup_commands=[], engine_command=f"{folder}mpiexec -n {{tasks}} {folder}abinit")
    assert references.environment_version("abinit", site) == "10.6.7" and references.environment_version("abinit", installed) == "10.0.3"
    assert references.release("abinit", "10.6.7") == "10.6.7" and references.release("abinit", "10.4.1") == "10.0.3"


def test_build_elk_checks_the_pinned_archive_and_indexes_it(pinned):
    built = references.build_elk("9.9.9", cache_dir=pinned.cache / "elk")
    assert built == {"engine": "elk", "version": "9.9.9", "records": 9, "examples": 2}
    payload = json.loads((references.root() / "elk-9.9.9.json").read_text(encoding="utf-8"))
    assert payload["sources"][0]["sha256"] == pinned.catalog["elk"]["releases"]["9.9.9"]["sha256"]
    # A changed archive is refused before anything is indexed.
    (pinned.cache / "elk" / "elk-9.9.9.tgz").write_bytes(b"tampered")
    with pytest.raises(AssertionError, match="cache"):
        references.build_elk("9.9.9", cache_dir=pinned.cache / "elk")


def test_abinit_variables_are_read_as_data_and_never_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    records = {record["id"]: record for record in references.parse_abinit_variables(VARIABLES)}
    assert not (tmp_path / "should-never-exist").exists()
    tsmear = records["keyword:tsmear"]
    assert (tsmear["title"], tsmear["type"], tsmear["default"], tsmear["varset"]) == ("tsmear: Temperature of SMEARing", "real", "0.01", "gstate")
    assert tsmear["unit"].startswith("Hartree, or eV") and tsmear["characteristics"] == ["ENERGY"] and tsmear["tags"] == ["Band Occ"]
    # Wiki links read as words, also across a line break and where the source left one unbalanced.
    assert tsmear["text"] == ("Broadening of the occupations of metals, for occopt 3 to 7. See Test2020 and the tutorial on "
                              "metals; set it with occopt.")
    ngkpt = records["keyword:ngkpt"]
    assert (ngkpt["default"], ngkpt["dimensions"], ngkpt["requires"], ngkpt["excludes"]) == ("3*0", "(3)", "kptopt >= 0", "kptrlatt")
    assert records["keyword:nspinor"]["default"] == "2 if pawspnorb == 1; otherwise 1"
    assert records["keyword:ecutsm"]["default"] == "0.5 Ha"
    # Basic variables rank first, developer ones last.
    assert ngkpt["prior"] > 0 > records["keyword:ecutsm"]["prior"] and records["keyword:tsmear"]["prior"] == 0
    gwgrid = records["keyword:gwgrid"]
    assert gwgrid["default"] == "__import__('os').getcwd()" and gwgrid["dimensions"] == "(1 to nkpt)"
    asr = references.parse_abinit_variables(ANADDB_VARIABLES, "anaddb")[0]
    assert (asr["id"], asr["name"], asr["program"], asr["title"]) == ("keyword:asr@anaddb", "asr", "anaddb", "asr (anaddb): Acoustic Sum Rule")
    assert asr["text"].endswith("see also asr for abinit.")


def test_abinit_tutorials_and_topics_read_as_plain_text():
    records = references.parse_abinit_markdown("doc/tutorial/tphon.md", TUTORIAL)
    assert [record["title"] for record in records] == ["Test tutorial on phonons: Phonons of a III-V semiconductor.",
                                                        "Test tutorial on phonons: 1 Interatomic force constants"]
    intro = records[0]["text"]
    assert "then ifcflag (anaddb).\n\nImportant:\n\n    Converge ecut first, see topic Phonons." in intro
    assert "(Input: tests/tutorespfn/Input/tphon_1.abi, tests/tutorespfn/Input/tphon_2.abi)" in intro
    assert "TUTORIAL_README" not in intro and "hidden remark" not in records[1]["text"] and records[0]["tags"] == ["tutorial", "tphon"]
    topic = {record["id"]: record for record in references.abinit_records(ABINIT_FILES) if record["source"] == "doc/topics/_Phonons.md"}
    first = topic["guide:doc/topics/_Phonons.md#topic-phonons"]
    assert first["title"] == "Topic Phonons" and first["summary"] == "How to compute phonon frequencies with this test package"
    # The variables that name the topic replace its template; empty template sections are dropped.
    assert topic["guide:doc/topics/_Phonons.md#related-input-variables"]["text"] == (
        "Basic: asr (anaddb)\nUseful: ifcflag (anaddb), ngqpt (anaddb), nqpath (anaddb), qpath (anaddb)")
    assert not any("selected" in identifier for identifier in topic)


def test_abinit_inputs_digest_features_and_chains():
    phonons = references.parse_abinit_input("tests/tutorespfn/Input/tphon_1.abi", PHONON_INPUT, NAMES)
    assert phonons["program"] == "abinit" and phonons["elements"] == ["Al", "As"] and phonons["chain"] == "tphon_1.abi, tphon_2.abi"
    assert {"abinit", "dfpt", "nc", "phonons", "dielectric"} <= set(phonons["tags"]) and not {"bands", "paw"} & set(phonons["tags"])
    assert phonons["summary"].startswith("Crystalline AlAs: phonons at all q points of a 4x4x4 mesh and the electric-field response.;")
    for fact in ("ndtset 2", "rfphon2 1", "rfelfd2 3", "kptopt2 3", "ngqpt 4 4 4", "2 atoms (Al, As)"):
        assert fact in phonons["summary"]
    assert "Crystalline AlAs: phonons;" not in phonons["summary"]   # the test description repeats the header
    assert "regression testing" not in phonons["text"] and "TEST_INFO" not in phonons["text"]
    # Dataset suffixes and words that are not variables are read correctly.
    assert phonons["variables"][:4] == ["ndtset", "rfphon", "rfelfd", "kptopt"] and "phonons" not in phonons["variables"]
    anaddb = references.parse_abinit_input("tests/tutorespfn/Input/tphon_2.abi", ANADDB_INPUT, NAMES)
    assert anaddb["program"] == "anaddb" and {"interatomic force constants", "phonon bands"} <= set(anaddb["tags"])
    assert "ifcflag 1" in anaddb["summary"] and "asr 1" in anaddb["summary"]
    gw = references.parse_abinit_input("tests/tutorial/Input/tgw_1.abi", PAW_INPUT, NAMES)
    assert {"gw", "paw"} <= set(gw["tags"])
    # A later step of a tutorial chain takes the material of the first.
    records = {record["id"]: record for record in references.abinit_records(ABINIT_FILES)}
    assert records["example:tests/tutorespfn/Input/tphon_2.abi"]["elements"] == ["Al", "As"]


def test_qe_input_descriptions_give_variables_units_and_cards():
    records = {record["id"]: record for record in references.parse_qe_helpdoc("PW/Doc/INPUT_PW.def", HELPDOC)}
    ecut = records["keyword:pw.x/&SYSTEM/ecutwfc"]
    assert (ecut["title"], ecut["type"], ecut["unit"], ecut["status"], ecut["dimensionality"]) == (
        "ecutwfc (pw.x &SYSTEM)", "real", "Ry", "REQUIRED", "energy")
    assert ecut["text"] == "kinetic energy cutoff for wavefunctions, see ecutrho"
    occupations = records["keyword:pw.x/&SYSTEM/occupations"]
    assert [choice["name"] for choice in occupations["choices"]] == ["'smearing'", "'fixed'"]
    assert occupations["choices"][0]["description"] == "gaussian smearing for metals; see degauss"
    assert records["keyword:pw.x/&SYSTEM/starting_magnetization"]["dimensions"] == "(1 to ntyp)"
    assert records["section:pw.x/&SYSTEM"]["keywords"] == ["degauss", "ecutwfc", "occupations", "starting_magnetization"]
    card = records["section:pw.x/K_POINTS"]
    assert [choice["name"] for choice in card["choices"]] == ["automatic", "gamma"] and card["default"] == "tpiba"
    assert card["keywords"] == ["nk1", "nk2", "nk3"] and "automatic: automatically generated uniform grid" in card["text"]
    nk1 = records["keyword:pw.x/K_POINTS/nk1"]
    assert (nk1["type"], nk1["text"]) == ("integer", "The k-point grid as in Monkhorst-Pack grids.")
    assert records["guide:PW/Doc/INPUT_PW.def#intro"]["text"] == "All quantities are in Rydberg atomic units unless stated."
    # pw.x ranks before the post-processing programs that share its names.
    assert ecut["prior"] > references.parse_qe_helpdoc("PP/Doc/INPUT_DOS.def", DOS_HELPDOC)[1]["prior"]


def test_qe_example_scripts_give_each_input_with_its_step():
    records = {record["id"]: record for record in references.qe_records(QE_FILES) if record["kind"] == "example"}
    scf = records["example:PHonon/examples/example02/run_example#alas.scf.in"]
    assert scf["program"] == "pw.x" and scf["elements"] == ["Al", "As"] and scf["source"] == "PHonon/examples/example02/run_example"
    assert scf["summary"].startswith("This example shows how to calculate phonon dispersions of AlAs.; 1) make a self-consistent calculation")
    for fact in ("calculation scf", "ecutwfc 16.0", "conv_thr 1.0d-8", "K_POINTS automatic 4 4 4 1 1 1", "2 atoms (Al, As)"):
        assert fact in scf["summary"]
    # The phonon step lists no species and takes those of the SCF of the same material.
    phonons = records["example:PHonon/examples/example02/run_example#alas.ph.in"]
    assert phonons["program"] == "ph.x" and phonons["elements"] == ["Al", "As"] and {"phonons", "dispersion"} <= set(phonons["tags"])
    assert "2) make a phonon calculation on a 4x4x4 grid of q-points" in phonons["summary"] and "ldisp .true." in phonons["summary"]
    md = records["example:PHonon/examples/example02/run_example#si.md.in"]
    assert "md" in md["tags"] and "running molecular dynamics of Si" in md["summary"] and md["elements"] == ["Si"]
    assert not any("plot.gnu" in identifier for identifier in records)
    epw = records["example:EPW/examples/test/epw/epw.in"]
    assert epw["program"] == "epw.x" and epw["elements"] == ["C"] and {"electron-phonon", "superconductivity", "wannier"} <= set(epw["tags"])


def test_qe_versions_come_from_modules_builds_and_the_login_node(pinned, monkeypatch):
    module = SimpleNamespace(setup_commands=["module load QuantumESPRESSO/7.6-gcc-2026.03-mpich-eb"], engine_command="", qe_bin_dir="")
    build = SimpleNamespace(setup_commands=[], engine_command="", qe_bin_dir="/work/me/software/qe-7.4.1-ab12cd/bin")
    assert references.environment_version("qe", module) == "7.6" and references.environment_version("qe", build) == "7.4.1"
    config = SimpleNamespace(engine="qe", host="hpc", user="me", remote_root="/work/me", setup_commands=["module load espresso"],
                             engine_command="", qe_bin_dir="/opt/qe/bin")
    cluster = Cluster("     Program PWSCF v.9.9 starts on 23Oct2026 at  0:20:48\n     Waiting for input...\n")
    monkeypatch.setattr(references, "_downloads", lambda name, cache_dir=None: pinned.cache / name)
    built = references.build_for_environment(config, cluster)
    assert built[0]["version"] == "9.9" and built[0]["examples"] == 5
    assert "timeout 60 /opt/qe/bin/pw.x < /dev/null" in cluster.commands[0] and "export OMP_NUM_THREADS=1" in cluster.commands[0]
    assert references.available("qe", config) == {"version": "9.9", "manual": True}


def test_qe_planner_prefers_pw_x_variables_and_marks_what_it_cannot_run(pinned, monkeypatch):
    monkeypatch.setattr(references, "_downloads", lambda name, cache_dir=None: pinned.cache / name)
    monkeypatch.delenv("DFT_AGENT_NO_REFERENCE_DOWNLOAD")
    excerpts = references.planner_context("qe", "Phonon dispersion of AlAs with degauss 0.02 Ry", elements=["Al", "As"])
    sources = [item["source"] for item in excerpts]
    assert "QE 9.9 pw.x input: &SYSTEM degauss" in sources and not any("dos.x" in source for source in sources)
    assert "QE 9.9 example: PHonon/examples/example02/run_example#alas.ph.in" in sources
    entry = references.read("qe", "keyword:pw.x/&SYSTEM/degauss")
    assert entry["url"] == "https://qe.invalid/Doc/INPUT_PW.html#degauss" and entry["unit"] == "Ry" and entry["license"] == "GPL-2.0-or-later"
    assert references.read("qe", "example:EPW/examples/test/epw/epw.in")["url"] == "https://github.invalid/QEF/q-e/blob/beef/EPW/examples/test/epw/epw.in"
    md = references.planner_context("qe", "Molecular dynamics of silicon", elements=["Si"])
    text = next(item["text"] for item in md if item["source"].endswith("si.md.in"))
    assert "Uses what DFT Agent's QE does not offer: molecular dynamics" in text


def test_qe_planner_sends_excerpts_and_records_their_sources(tmp_path, monkeypatch):
    from importlib.resources import files

    from test_qe_agent import response
    from vasp_slurm_agent import qe_agent
    from vasp_slurm_agent.config import ClusterConfig
    source = tmp_path / "Si.cif"
    source.write_bytes(files("vasp_slurm_agent").joinpath("examples", "Si.cif").read_bytes())
    config = ClusterConfig(host="example.invalid", user="test", remote_root="/scratch/test", partition="cpu", engine="qe",
                           qe_pseudo_root="/pseudo", qe_pseudopotentials={"Si": "Si.UPF"})
    seen = []

    def request(payload, schema, settings, **kwargs):
        seen.append((json.loads(payload), kwargs["instructions"]))
        return json.dumps(response(("scf",)))
    monkeypatch.setattr(agent, "_request_structured", request)
    excerpt = {"source": "QE 7.6 pw.x input: &SYSTEM degauss", "text": "degauss: value of the gaussian spreading in metals"}
    monkeypatch.setattr(references, "planner_context", lambda engine, goal, **kwargs: [excerpt] if engine == "qe" else [])
    plan = qe_agent.draft_task("SCF with degauss 0.02 Ry", source, agent.ModelSettings("codex"), config=config)
    assert seen[0][0]["references"] == [excerpt] and "references, when present" in seen[0][1]
    assert plan["references"] == ["QE 7.6 pw.x input: &SYSTEM degauss"]
    qe_agent.draft_task("SCF", source, agent.ModelSettings("codex"), config=config, references=False)
    assert "references" not in seen[1][0]


def test_qe_native_options_are_checked_against_the_input_description(pinned, monkeypatch, tmp_path):
    from importlib.resources import files

    from test_qe_agent import response
    from vasp_slurm_agent import qe_agent
    from vasp_slurm_agent.config import ClusterConfig
    monkeypatch.setattr(references, "_downloads", lambda name, cache_dir=None: pinned.cache / name)
    config = ClusterConfig(host="example.invalid", user="test", remote_root="/scratch/test", partition="cpu", engine="qe",
                           setup_commands=["module load QuantumESPRESSO/9.9-gcc"], qe_pseudo_root="/pseudo",
                           qe_pseudopotentials={"Si": "Si.UPF"})
    # Without a built index nothing is checked.
    assert references.qe_namelist_problems({"system": {"degaus": 0.02}}, config) == []
    references.build_qe("9.9", cache_dir=pinned.cache / "qe-docs")
    assert references.qe_namelist_problems({"system": {"degauss": 0.02, "starting_magnetization(1)": 0.5},
                                            "electrons": {"mixing_beta": 0.3, "conv_thr": 1e-8}}, config) == []
    assert references.qe_namelist_problems({"system": {"degaus": 0.02, "mixing_beta": 0.3}}, config) == [
        "QE 9.9's pw.x has no &SYSTEM variable degaus.", "QE 9.9's pw.x has no &SYSTEM variable mixing_beta; it belongs to &ELECTRONS."]
    # Sections DFT Agent does not map, and namelists the index does not describe, are left to QE.
    assert references.qe_namelist_problems({"inputepw": {"anything": 1}, "inputph": {"tr2_ph": 1e-14}}, config) == []
    source = tmp_path / "Si.cif"
    source.write_bytes(files("vasp_slurm_agent").joinpath("examples", "Si.cif").read_bytes())
    monkeypatch.setattr(agent, "_request_structured", lambda *args, **kwargs: json.dumps(
        response(("scf",), {"namelists": {"system": {"mixing_beta": 0.3}}})))
    monkeypatch.setattr(references, "planner_context", lambda *args, **kwargs: [])
    plan = qe_agent.draft_task("SCF with mixing_beta 0.3", source, agent.ModelSettings("codex"), config=config)
    assert plan["status"] == "needs_input" and plan["questions"] == [
        "QE 9.9's pw.x has no &SYSTEM variable mixing_beta; it belongs to &ELECTRONS."]


def test_fallback_manual_does_not_reject_new_or_unknown_qe_versions(pinned, monkeypatch):
    monkeypatch.setattr(references, "_downloads", lambda name, cache_dir=None: pinned.cache / name)
    references.build_qe("9.9", cache_dir=pinned.cache / "qe-docs")
    newer = SimpleNamespace(setup_commands=["module load QuantumESPRESSO/10.1"], engine_command="", qe_bin_dir="")
    assert references.qe_namelist_problems({"system": {"new_setting": True}}, newer) == []
    assert references.qe_namelist_problems({"system": {"new_setting": True}}) == []
    excerpts = references.planner_context("qe", "degauss 0.02 Ry", config=newer)
    assert excerpts and all("No manual for QE 10.1 is pinned" in item["notes"][0] for item in excerpts)


def test_reference_batch_shares_one_download_deadline(monkeypatch, tmp_path):
    from itertools import count
    ticks, fetched = count(), []
    monkeypatch.setattr(references.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(references, "_fetch", lambda item, cache, timeout: fetched.append(timeout))
    with pytest.raises(TimeoutError, match="cached files can be reused"):
        references._fetch_many(list(range(20)), tmp_path, timeout=3)
    assert len(fetched) <= 2 and all(0 < value <= 3 for value in fetched)


def test_qe_reference_checks_the_configured_binary_and_binding(pinned, monkeypatch):
    monkeypatch.setattr(references, "_downloads", lambda name, cache_dir=None: pinned.cache / name)
    config = SimpleNamespace(engine="qe", host="hpc", user="me", setup_commands=[], engine_command="", qe_bin_dir="",
                             qe_command="srun /opt/qe-custom/bin/pw.x")
    cluster = Cluster("Program PWSCF v.9.9 starts\n")
    references.build_for_environment(config, cluster)
    assert "timeout 60 /opt/qe-custom/bin/pw.x" in cluster.commands[0]
    assert references.environment_version("qe", config) == "9.9"
    config.qe_command = "srun /opt/another-build/bin/pw.x"
    assert references.environment_version("qe", config) is None


def test_build_abinit_checks_each_pinned_file(pinned):
    built = references.build_abinit("9.9.9", cache_dir=pinned.cache / "abinit-docs")
    payload = json.loads((references.root() / "abinit-9.9.9.json").read_text(encoding="utf-8"))
    assert built == {"engine": "abinit", "version": "9.9.9", "records": len(payload["records"]), "examples": 3}
    assert {record["kind"] for record in payload["records"]} == {"keyword", "guide", "example"}
    assert payload["sources"] == [{"url": "https://raw.invalid/abinit/c0ffee/", "files": 7}]
    assert payload["license"]["name"] == "GPL-3.0-only"
    (pinned.cache / "abinit-docs" / "9.9.9" / "doc__topics___Phonons.md").write_text("tampered")
    with pytest.raises(AssertionError, match="cache"):
        references.build_abinit("9.9.9", cache_dir=pinned.cache / "abinit-docs")


def test_cp2k_keywords_are_indexed_once_under_their_central_path():
    version, records = references.parse_cp2k_xml(XML)
    assert version == "2026.2"
    by_id = {record["id"]: record for record in records}
    eps = by_id["keyword:FORCE_EVAL/SCF/EPS_SCF"]
    assert eps["also_in"] == 1 and "keyword:ATOM/SCF/EPS_SCF" not in by_id
    assert (eps["type"], eps["default"], eps["usage"]) == ("real", "1.0E-005", "EPS_SCF 1.e-6")
    preconditioner = by_id["keyword:FORCE_EVAL/SCF/OT/PRECONDITIONER"]
    assert preconditioner["aliases"] == ["PRECOND"] and [choice["name"] for choice in preconditioner["choices"]] == ["FULL_ALL", "FULL_KINETIC"]
    assert by_id["section:FORCE_EVAL/SCF"]["text"].endswith("See OT (FORCE_EVAL/SCF/OT).")
    assert by_id["section:FORCE_EVAL/SCF/OT"]["parameters"]["lone"] == "T"
    assert by_id["keyword:MOTION/MD/TIMESTEP"]["unit"] == "fs"
    assert by_id["keyword:MOTION/MD/PRINT/TIMESTEP"]["type"] == "integer"
    with pytest.raises(ValueError, match="input reference"):
        references.parse_cp2k_xml(b'<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "b">]><CP2K_INPUT/>')
    payload = references.cp2k_input_payload(gzip.compress(XML))
    assert payload["version"] == "2026.2" and payload["part"] == "keywords"


def test_cp2k_input_digest_elements_and_features():
    record = references.parse_cp2k_input("benchmarks/QS/water.inp", CP2K_INPUT)
    assert record["elements"] == ["H", "O"] and {"md", "ot"} <= set(record["tags"]) and "kpoints" not in record["tags"]
    assert record["summary"].startswith("A small water box under a thermostat.")
    for fact in ("RUN_TYPE MD", "ENSEMBLE NVT", "TIMESTEP 0.5", "THERMOSTAT CSVR", "CUTOFF 400", "XC_FUNCTIONAL PBE", "9 atoms (H, O)"):
        assert fact in record["summary"]
    # Long coordinate lists are shortened in the stored text.
    assert record["text"].count(" 0.0 0.0") == 6 and "..." in record["text"]


def test_cp2k_markdown_is_split_at_headings_outside_code():
    records = references.parse_cp2k_markdown("docs/methods/sampling/molecular_dynamics.md", MARKDOWN)
    assert [record["title"] for record in records] == ["Molecular Dynamics", "Molecular Dynamics: Thermostats",
                                                        "Molecular Dynamics: Equilibration"]
    thermostats = records[1]
    assert "CSVR (MOTION/MD/THERMOSTAT/CSVR)" in thermostats["text"] and "## not a heading" in thermostats["text"]
    assert "youtube" not in records[0]["text"] and records[0]["tags"] == ["methods", "sampling"]
    assert thermostats["id"] == "guide:docs/methods/sampling/molecular_dynamics.md#thermostats"


def test_cp2k_documents_build_from_pinned_files(pinned):
    built = references.build_cp2k_documents(cache_dir=pinned.cache / "cp2k-docs")
    assert built == {"engine": "cp2k", "part": "documents", "version": "2026.2", "records": 4, "examples": 1}
    assert (references.root() / "cp2k-docs-v2026.2.json").is_file()


class Cluster:
    """A login node that answers the commands a reference build sends."""

    def __init__(self, output=""):
        self.commands, self.downloads, self.output = [], [], output

    def run(self, command, timeout=60):
        self.commands.append(command)
        return Result(0, self.output, "")

    def download(self, remote, local):
        self.downloads.append(remote)
        local.write_bytes(gzip.compress(XML))
        return {}

    def close(self):
        pass


def test_cp2k_keywords_come_from_the_installed_program(pinned):
    config = SimpleNamespace(engine="cp2k", host="hpc", user="me", remote_root="/work/me",
                             setup_commands=["module load CP2K/2026.2-gcc-2026.03-mpich-eb"],
                             engine_command="mpiexec -n {tasks} /opt/cp2k/bin/cp2k.psmp")
    cluster = Cluster()
    built = references.build_cp2k_keywords(config, cluster)
    assert built == {"engine": "cp2k", "part": "keywords", "version": "2026.2", "records": 11}
    command = cluster.commands[0]
    assert command.splitlines()[:2] == ["set -e", "module load CP2K/2026.2-gcc-2026.03-mpich-eb"]
    assert "/opt/cp2k/bin/cp2k.psmp --xml" in command and "gzip -f cp2k_input.xml" in command and "mpiexec" not in command
    assert len(cluster.downloads) == 1
    assert re.fullmatch(r"/work/me/\.dft-agent/reference-[0-9a-f]{32}/cp2k_input.xml.gz", cluster.downloads[0])
    assert "export OMP_NUM_THREADS=1" in cluster.commands[0]
    assert cluster.commands[1].startswith("rm -f " + cluster.downloads[0])
    references.build_cp2k_keywords(config, cluster)
    assert cluster.downloads[0] != cluster.downloads[1]
    assert references.available("cp2k", config) == {"version": "2026.2", "documents": False, "keywords": True}
    with pytest.raises(ValueError, match="cp2k.psmp"):
        references.build_cp2k_keywords(SimpleNamespace(**{**vars(config), "engine_command": "srun vasp_std"}), cluster)


def test_elk_version_is_read_on_the_login_node_when_the_module_does_not_say(pinned, monkeypatch):
    config = SimpleNamespace(engine="elk", host="hpc", user="me", remote_root="/work/me", setup_commands=["module load elk"],
                             engine_command="srun elk")
    cluster = Cluster("┌──────┐\n│ Elk code version 9.9.9 started │\n\nError(readinput): error opening elk.in\n")
    monkeypatch.setattr(references, "_downloads", lambda name, cache_dir=None: pinned.cache / name)
    built = references.build_for_environment(config, cluster)
    assert built[0]["version"] == "9.9.9" and built[0]["environment_version"] == "9.9.9"
    assert "timeout 60 elk < /dev/null" in cluster.commands[0] and 'rm -rf "$folder"' in cluster.commands[0]
    # The check is remembered for this environment.
    assert references.environment_version("elk", config) == "9.9.9"


def test_abinit_version_is_read_on_the_login_node_when_the_module_does_not_say(pinned, monkeypatch):
    config = SimpleNamespace(engine="abinit", host="hpc", user="me", remote_root="/work/me", setup_commands=["module load abinit"],
                             engine_command="srun abinit")
    cluster = Cluster("- OMP_NUM_THREADS was not defined. It has been set to 1.\n9.9.9\n")
    monkeypatch.setattr(references, "_downloads", lambda name, cache_dir=None: pinned.cache / name)
    built = references.build_for_environment(config, cluster)
    assert built[0]["version"] == "9.9.9" and built[0]["examples"] == 3
    assert "timeout 60 abinit --version" in cluster.commands[0] and references.environment_version("abinit", config) == "9.9.9"
    assert references.available("abinit", config) == {"version": "9.9.9", "manual": True}


def test_abinit_planner_reads_named_variables_and_marks_what_it_cannot_run(pinned, monkeypatch):
    monkeypatch.setattr(references, "_downloads", lambda name, cache_dir=None: pinned.cache / name)
    monkeypatch.delenv("DFT_AGENT_NO_REFERENCE_DOWNLOAD")
    reference = references.load("abinit")
    assert reference.search("What does tsmear do?", limit=1)[0][0]["id"] == "keyword:tsmear"
    assert reference.find("asr")["id"] == "keyword:asr@anaddb"
    entry = references.read("abinit", "tsmear")
    assert entry["source"] == "ABINIT 9.9.9 input variable tsmear" and entry["url"] == "https://docs.abinit.invalid/variables/gstate/#tsmear"
    assert entry["license"] == "GPL-3.0-only" and entry["unit"].startswith("Hartree")
    assert references.read("abinit", "asr@anaddb")["source"] == "ABINIT 9.9.9 anaddb input variable asr"
    assert references.read("abinit", "example:tests/tutorespfn/Input/tphon_1.abi")["url"] == (
        "https://github.invalid/abinit/abinit/blob/c0ffee/tests/tutorespfn/Input/tphon_1.abi")
    assert references.read("abinit", "guide:doc/topics/_Phonons.md#introduction")["url"] == "https://docs.abinit.invalid/topics/Phonons/"
    excerpts = references.planner_context("abinit", "Phonon bands and dielectric constant of AlAs", elements=["Al", "As"])
    sources = [item["source"] for item in excerpts]
    assert set(sources[:2]) == {"ABINIT 9.9.9 example: tests/tutorespfn/Input/tphon_1.abi", "ABINIT 9.9.9 example: tests/tutorespfn/Input/tphon_2.abi"}
    # Variables only when the user names them.
    assert not any("input variable" in source for source in sources)
    named = references.planner_context("abinit", "Keep tsmear at 0.01 for AlAs", elements=["Al", "As"])
    assert "ABINIT 9.9.9 input variable tsmear" in [item["source"] for item in named]
    gw = references.planner_context("abinit", "GW band gap of AlAs with PAW", elements=["Al", "As"])
    text = next(item["text"] for item in gw if item["source"].endswith("tgw_1.abi"))
    assert "Uses what DFT Agent's ABINIT does not offer: PAW, GW" in text and "# Aluminium arsenide" not in text


def test_search_ranks_names_examples_and_chinese_requests(pinned, monkeypatch):
    monkeypatch.setattr(references, "_downloads", lambda name, cache_dir=None: pinned.cache / name)
    monkeypatch.delenv("DFT_AGENT_NO_REFERENCE_DOWNLOAD")
    reference = references.load("elk")
    assert reference.parts["manual"]["version"] == "9.9.9" and reference.notes == []
    hits = reference.search("What does lradstp do?", limit=3)
    assert hits[0][0]["id"] == "keyword:lradstp"
    hits = reference.search("计算 LiF 的光学吸收谱和激子", elements=["Li", "F"], limit=4)
    assert hits[0][0]["id"] in {"example:TDDFT-optics/LiF-bootstrap", "example:BSE/LiF"}
    assert {record["kind"] for record, _ in reference.search("lithium", kinds=["lesson"])} == {"lesson"}
    assert reference.find("task 32")["id"] == "task:31/2/3" and reference.find("ngrkf")["id"] == "keyword:wplot"
    entry = references.read("elk", "lradstp")
    assert entry["default"] == "4" and entry["license"] == "GPL-3.0-or-later" and entry["source"] == "Elk 9.9.9 manual, input block lradstp"
    assert references.read("elk", "example:BSE/LiF")["note"].startswith("An official demonstration")


def test_planner_excerpts_prefer_runnable_examples_and_mark_the_rest(pinned, monkeypatch):
    monkeypatch.setattr(references, "_downloads", lambda name, cache_dir=None: pinned.cache / name)
    monkeypatch.delenv("DFT_AGENT_NO_REFERENCE_DOWNLOAD")
    excerpts = references.planner_context("elk", "Optical absorption of LiF with excitons", elements=["Li", "F"])
    sources = [item["source"] for item in excerpts]
    assert sources.index("Elk 9.9.9 example TDDFT-optics/LiF-bootstrap") < sources.index("Elk 9.9.9 example BSE/LiF")
    bse = next(item["text"] for item in excerpts if item["source"].endswith("BSE/LiF"))
    assert "Not available in DFT Agent's Elk tasks: 185, 186" in bse
    runnable = next(item["text"] for item in excerpts if item["source"].endswith("LiF-bootstrap"))
    # The description is given once, above the input; comments on value lines stay.
    assert "Tasks: 0 (Ground state from atomic densities.)" in runnable and "! Optical spectrum" not in runnable
    assert "0         ! ground state" in runnable
    assert all(len(item["text"]) <= 1400 for item in excerpts)
    # Input names reach the planner only when the user wrote them or they are the best match.
    assert not any("input block" in source for source in sources)
    assert any("input block lradstp" in item["source"] for item in references.planner_context("elk", "Why is lradstp 2 here?"))


def test_cp2k_search_puts_central_paths_and_named_keywords_first(pinned, monkeypatch):
    monkeypatch.setattr(references, "_downloads", lambda name, cache_dir=None: pinned.cache / name)
    monkeypatch.delenv("DFT_AGENT_NO_REFERENCE_DOWNLOAD")
    payload = references.cp2k_input_payload(gzip.compress(XML))
    references._write(references.root() / "cp2k-input-2026.2.json", payload)
    config = SimpleNamespace(engine="cp2k", setup_commands=["module load CP2K/2026.2-x"], engine_command="srun cp2k.psmp")
    reference = references.load("cp2k", config)
    assert set(reference.parts) == {"documents", "keywords"} and reference.notes == []
    assert reference.search("time step of molecular dynamics", kinds=["keyword"])[0][0]["id"] == "keyword:MOTION/MD/TIMESTEP"
    assert reference.find("FORCE_EVAL.SCF.OT.PRECONDITIONER")["name"] == "PRECONDITIONER"
    excerpts = references.planner_context("cp2k", "AIMD of liquid water at 300 K, check EPS_SCF", elements=["H", "O"], config=config)
    sources = [item["source"] for item in excerpts]
    assert "DFT Agent cluster check: Equilibrate liquid water first" in sources
    assert "CP2K 2026.2 input reference (cp2k --xml): FORCE_EVAL/SCF/EPS_SCF" in sources
    assert "CP2K 2026.2 example: benchmarks/QS/water.inp" in sources
    # Another environment's version is used with a note when its own was not read.
    other = SimpleNamespace(engine="cp2k", setup_commands=["module load CP2K/2025.1"], engine_command="srun cp2k.psmp")
    assert "this environment runs CP2K 2025.1" in references.load("cp2k", other).notes[0]


def test_missing_references_never_stop_planning():
    # Tests run without downloads: only the bundled lessons are available, with notes.
    elk = references.load("elk")
    assert {record["kind"] for record in elk.records} == {"lesson"} and elk.parts == {}
    cp2k = references.load("cp2k")
    assert "keyword reference is not built yet" in cp2k.notes[-1]
    assert references.available("elk") == {"version": references.catalog()["elk"]["default"], "manual": False}
    abinit = references.load("abinit")
    assert {record["kind"] for record in abinit.records} == {"lesson"} and abinit.parts == {}
    qe = references.load("qe")
    assert {record["kind"] for record in qe.records} == {"lesson"} and qe.parts == {}
    assert references.planner_context("abinit", "Phonons of silicon", elements=["Si"])[0]["source"].startswith("DFT Agent cluster check")


def test_engine_planner_sends_excerpts_and_records_their_sources(tmp_path, monkeypatch):
    from test_engine_adapters import DATA, _plan
    source = tmp_path / "LiF.cif"
    shutil.copyfile(DATA / "Si-perturbed.cif", source)
    seen = []

    def request(payload, schema, settings, **kwargs):
        seen.append((json.loads(payload), kwargs["instructions"]))
        return json.dumps(_plan(["scf"], {"mesh": [2, 2, 2]}))
    monkeypatch.setattr(agent, "_request_structured", request)
    excerpt = {"source": "Elk 11.0.2 example TDDFT-optics/LiF-bootstrap", "text": "tasks 0 120 121 320"}
    monkeypatch.setattr(engine_agent, "planner_context", lambda engine, goal, **kwargs: [excerpt])
    plan = engine_agent.draft_task("Optical spectrum with excitons", source, agent.ModelSettings("codex"), engine="elk")
    assert seen[0][0]["references"] == [excerpt] and "never put their native input names" in seen[0][1]
    assert plan["references"] == ["Elk 11.0.2 example TDDFT-optics/LiF-bootstrap"]
    engine_agent.draft_task("SCF", source, agent.ModelSettings("codex"), engine="elk", references=False)
    engine_agent.draft_task("SCF", source, agent.ModelSettings("codex"), engine="abinit")
    assert "references" not in seen[1][0] and seen[2][0]["references"] == [excerpt]


def test_mcp_reference_tools(tmp_path, monkeypatch):
    from test_agent_mode import call, tools_for
    from vasp_slurm_agent import credentials
    from vasp_slurm_agent.connections import ConnectionRegistry
    from vasp_slurm_agent.workspace import Workspace

    store = {}
    monkeypatch.setattr(credentials, "_keyring", lambda: SimpleNamespace(
        set_password=lambda s, n, v: store.__setitem__((s, n), v), get_password=lambda s, n: store.get((s, n)),
        delete_password=lambda s, n: store.pop((s, n), None)))
    workspace = Workspace(tmp_path / "home")
    registry = ConnectionRegistry(workspace)
    pid = workspace.create_project("LiF")["id"]
    project = SimpleNamespace(workspace=workspace, registry=registry, pid=pid, sid=workspace.create_session(pid)["id"],
                              model=registry.create("model", "Codex", {"provider": "codex"})["id"])
    hpc = registry.create("hpc", "Cluster", {"host": "hpc.invalid", "user": "me", "remote_base": "/work/me", "scheduler": "slurm"})
    registry.bind(pid, hpc["id"])
    elk = registry.create_environment(hpc["id"], "Elk", "elk", {"engine_command": "srun elk", "partition": "cpu",
                                                               "engine_library_root": "/work/me/library/elk",
                                                               "setup_commands": ["module load Elk/11.0.2-gcc"]})
    tools = tools_for(project)
    names = {item["name"] for item in tools.definitions()}
    assert {"search_reference", "read_reference", "build_reference"} <= names
    seen = {}
    monkeypatch.setattr(references, "search", lambda engine, query, **kwargs: seen.update(engine=engine, query=query, **kwargs) or {"results": []})
    error, result = call(tools, "search_reference", engine="elk", query="bootstrap kernel", kind="example", environment_id=elk["id"])
    assert not error and seen["kinds"] == ["example"] and references.environment_version("elk", seen["config"]) == "11.0.2"
    error, text = call(tools, "search_reference", engine="cp2k", query="OT", environment_id=elk["id"])
    assert error and "runs elk, not cp2k" in text
    error, text = call(tools, "search_reference", engine="vasp", query="ENCUT")
    assert error
    error, result = call(tools, "search_reference", engine="abinit", query="tsmear")
    assert not error and seen["engine"] == "abinit" and seen["config"] is None
    error, overview = call(tools, "cluster_status")
    assert overview["clusters"][0]["environments"][0]["reference"] == {"version": "11.0.2", "manual": False}
    monkeypatch.setattr(references, "build_for_environment", lambda config, transport=None: [{"engine": "elk", "version": "11.0.2"}])
    error, built = call(tools, "build_reference", environment_id=elk["id"])
    assert not error and built["built"] == [{"engine": "elk", "version": "11.0.2"}] and built["environment_version"] == "11.0.2"


def test_software_panel_offers_the_manuals_and_shows_them_once_built(monkeypatch):
    from streamlit.testing.v1 import AppTest
    from vasp_slurm_agent import credentials
    from vasp_slurm_agent.connections import ConnectionRegistry

    store = {}
    monkeypatch.setattr(credentials, "_keyring", lambda: SimpleNamespace(
        set_password=lambda s, n, v: store.__setitem__((s, n), v), get_password=lambda s, n: store.get((s, n)),
        delete_password=lambda s, n: store.pop((s, n), None)))
    registry = ConnectionRegistry()
    hpc = registry.create("hpc", "Cluster", {"host": "hpc.invalid", "user": "me", "remote_base": "/work/me", "scheduler": "slurm"})
    elk = registry.create_environment(hpc["id"], "Elk", "elk", {"engine_command": "srun elk", "partition": "cpu",
                                                               "engine_library_root": "/work/me/library/elk",
                                                               "setup_commands": ["module load Elk/11.0.2-gcc"]})
    registry.create_environment(hpc["id"], "VASP", "vasp", {"vasp_command": "srun vasp_std", "potcar_root": "/licensed", "partition": "cpu"})
    script = f"""
from vasp_slurm_agent.connections import ConnectionRegistry
from vasp_slurm_agent.setup_app import _reference_panel
registry = ConnectionRegistry()
_reference_panel(registry, registry.environments({hpc['id']!r}), "panel_")
"""
    view = AppTest.from_string(script).run()
    assert [button.label for button in view.button] == ["Get manuals and examples"] and not view.caption
    built = []
    monkeypatch.setattr(ConnectionRegistry, "build_reference", lambda self, identifier: built.append(identifier) or {})
    view.button[0].click().run()
    assert built == [elk["id"]] and not view.error
    references._write(references.root() / "elk-11.0.2.json", {"schema_version": 1, "engine": "elk", "part": "manual",
                                                             "version": "11.0.2", "records": []})
    view = AppTest.from_string(script).run()
    assert not view.button and [caption.value for caption in view.caption] == ["Manual and examples ready: Elk 11.0.2"]
