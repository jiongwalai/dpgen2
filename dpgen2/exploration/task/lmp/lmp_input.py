import random
from typing import (
    List,
    Optional,
)

import dpdata
import numpy as np
import scipy.constants as pc
from packaging.version import (
    Version,
)

from dpgen2.constants import (
    lmp_model_devi_name,
    lmp_pimd_model_devi_name,
    lmp_pimd_traj_name,
    lmp_traj_name,
)


def _sample_sphere():
    while True:
        vv = np.array([np.random.normal(), np.random.normal(), np.random.normal()])
        vn = np.linalg.norm(vv)
        if vn < 0.2:
            continue
        return vv / vn


def make_lmp_input(
    conf_file: str,
    ensemble: str,
    graphs: List[str],
    nsteps: int,
    dt: float,
    neidelay: Optional[int],
    trj_freq: int,
    mass_map: List[float],
    temp: float,
    tau_t: float = 0.1,
    pres: Optional[float] = None,
    tau_p: float = 0.5,
    use_clusters: bool = False,
    relative_f_epsilon: Optional[float] = None,
    relative_v_epsilon: Optional[float] = None,
    pka_e: Optional[float] = None,
    ele_temp_f: Optional[float] = None,
    ele_temp_a: Optional[float] = None,
    nopbc: bool = False,
    max_seed: int = 1000000,
    deepmd_version="2.0",
    nvnmd_version=None,
    trj_seperate_files=True,
    pimd_bead: Optional[str] = None,
):
    if (ele_temp_f is not None or ele_temp_a is not None) and Version(
        deepmd_version
    ) < Version("1"):
        raise RuntimeError(
            "the electron temperature is only supported by deepmd-kit >= 1.0.0, please upgrade your deepmd-kit"
        )
    if ele_temp_f is not None and ele_temp_a is not None:
        raise RuntimeError(
            "the frame style ele_temp and atom style ele_temp should not be set at the same time"
        )
    if "npt" in ensemble and pres is None:
        raise RuntimeError("the pressre should be provided for npt ensemble")
    ret = "variable        NSTEPS          equal %d\n" % nsteps
    ret += "variable        THERMO_FREQ     equal %d\n" % trj_freq
    ret += "variable        DUMP_FREQ       equal %d\n" % trj_freq
    ret += "variable        TEMP            equal %f\n" % temp
    if ele_temp_f is not None and nvnmd_version is None:
        ret += "variable        ELE_TEMP        equal %f\n" % ele_temp_f
    if ele_temp_a is not None and nvnmd_version is None:
        ret += "variable        ELE_TEMP        equal %f\n" % ele_temp_a
    if pres is not None:
        ret += "variable        PRES            equal %f\n" % pres
    ret += "variable        TAU_T           equal %f\n" % tau_t
    if pres is not None:
        ret += "variable        TAU_P           equal %f\n" % tau_p
    ret += "\n"
    ret += "units           metal\n"
    if nopbc:
        ret += "boundary        f f f\n"
    else:
        ret += "boundary        p p p\n"
    ret += "atom_style      atomic\n"
    ret += "\n"
    ret += "neighbor        1.0 bin\n"
    if neidelay is not None:
        ret += "neigh_modify    delay %d\n" % neidelay
    ret += "\n"
    ret += "box          tilt large\n"
    ret += (
        'if "${restart} > 0" then "read_restart dpgen.restart.*" else "read_data %s"\n'
        % conf_file
    )
    ret += "change_box   all triclinic\n"
    for jj in range(len(mass_map)):
        ret += "mass            %d %f\n" % (jj + 1, mass_map[jj])
    graph_list = ""
    for ii in graphs:
        graph_list += ii + " "
    model_devi_file_name = (
        lmp_pimd_model_devi_name % pimd_bead
        if pimd_bead is not None
        else lmp_model_devi_name
    )
    if Version(deepmd_version) < Version("1") and nvnmd_version is None:
        # 0.x
        ret += "pair_style      deepmd %s ${THERMO_FREQ} %s\n" % (
            graph_list,
            model_devi_file_name,
        )
    elif nvnmd_version is not None:
        ret += "pair_style      nvnmd %s\n" % ("model.pb")
    else:
        # 1.x
        keywords = ""
        if use_clusters:
            keywords += "atomic "
        if relative_f_epsilon is not None:
            keywords += "relative %s " % relative_f_epsilon
        if relative_v_epsilon is not None:
            keywords += "relative_v %s " % relative_v_epsilon
        if ele_temp_f is not None:
            keywords += "fparam ${ELE_TEMP}"
        if ele_temp_a is not None:
            keywords += "aparam ${ELE_TEMP}"
        ret += "pair_style      deepmd %s out_freq ${THERMO_FREQ} out_file %s %s\n" % (
            graph_list,
            model_devi_file_name,
            keywords,
        )
    ret += "pair_coeff      * *\n"
    ret += "\n"
    ret += "thermo_style    custom step temp pe ke etotal press vol lx ly lz xy xz yz\n"
    ret += "thermo          ${THERMO_FREQ}\n"
    if trj_seperate_files:
        if nvnmd_version is None:
            ret += "dump            1 all custom ${DUMP_FREQ} traj/*.lammpstrj id type x y z fx fy fz\n"
        else:
            ret += "dump            1 all custom ${DUMP_FREQ} ${rerun}_traj/*.lammpstrj id type x y z fx fy fz\n"
    else:
        lmp_traj_file_name = (
            lmp_pimd_traj_name % pimd_bead if pimd_bead is not None else lmp_traj_name
        )
        if nvnmd_version is None:
            ret += (
                "dump            1 all custom ${DUMP_FREQ} %s id type x y z fx fy fz\n"
                % lmp_traj_file_name
            )
        else:
            ret += (
                "dump            1 all custom ${DUMP_FREQ} ${rerun}_%s id type x y z fx fy fz\n"
                % lmp_traj_file_name
            )
    ret += "restart         10000 dpgen.restart\n"
    ret += "\n"
    if nvnmd_version is not None:
        ret += 'if "${rerun} > 0" then "jump SELF rerun"\n'
    if pka_e is None:
        ret += 'if "${restart} == 0" then "velocity        all create ${TEMP} %d"' % (
            random.randrange(max_seed - 1) + 1
        )
    else:
        sys = dpdata.System(conf_file, fmt="lammps/lmp")
        sys_data = sys.data
        pka_mass = mass_map[sys_data["atom_types"][0] - 1]
        pka_vn = (
            pka_e
            * pc.electron_volt
            / (0.5 * pka_mass * 1e-3 / pc.Avogadro * (pc.angstrom / pc.pico) ** 2)
        )  # type: ignore
        pka_vn = np.sqrt(pka_vn)
        print(pka_vn)
        pka_vec = _sample_sphere()
        pka_vec *= pka_vn
        ret += "group           first id 1\n"
        ret += 'if "${restart} == 0" then "velocity        first set %f %f %f"\n' % (
            pka_vec[0],
            pka_vec[1],
            pka_vec[2],
        )
        ret += "fix	       2 all momentum 1 linear 1 1 1\n"
    ret += "\n"
    if ensemble.split("-")[0] == "npt":
        assert pres is not None
        if nopbc:
            raise RuntimeError("ensemble %s is conflicting with nopbc" % ensemble)
    if ensemble == "npt" or ensemble == "npt-i" or ensemble == "npt-iso":
        ret += "fix             1 all npt temp ${TEMP} ${TEMP} ${TAU_T} iso ${PRES} ${PRES} ${TAU_P}\n"
    elif ensemble == "npt-a" or ensemble == "npt-aniso":
        ret += "fix             1 all npt temp ${TEMP} ${TEMP} ${TAU_T} aniso ${PRES} ${PRES} ${TAU_P}\n"
    elif ensemble == "npt-t" or ensemble == "npt-tri":
        ret += "fix             1 all npt temp ${TEMP} ${TEMP} ${TAU_T} tri ${PRES} ${PRES} ${TAU_P}\n"
    elif ensemble == "nvt":
        ret += "fix             1 all nvt temp ${TEMP} ${TEMP} ${TAU_T}\n"
    elif ensemble == "nve":
        ret += "fix             1 all nve\n"
    else:
        raise RuntimeError("unknown emsemble " + ensemble)
    if nopbc:
        ret += "velocity        all zero linear\n"
        ret += "fix             fm all momentum 1 linear 1 1 1\n"
    ret += "\n"
    ret += "timestep        %f\n" % dt
    ret += "run             ${NSTEPS} upto\n"
    if nvnmd_version is not None:
        ret += "jump SELF end\n"
        ret += "label rerun\n"
        if trj_seperate_files:
            ret += "rerun 0_traj/*.lammpstrj dump x y z fx fy fz add yes\n"
        else:
            ret += "rerun 0_%s dump x y z fx fy fz add yes\n" % lmp_traj_name
        ret += "label end\n"
    return ret
