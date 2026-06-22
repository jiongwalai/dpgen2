"""Prep and Run DeepMD tasks."""
import os
from pathlib import (
    Path,
)
from typing import (
    Any,
    List,
    Optional,
    Tuple,
)

import dpdata
import numpy as np
from dargs import (
    Argument,
    dargs,
)
from dflow.python import (
    FatalError,
    OPIO,
    TransientError,
)

from dpgen2.constants import (
    fp_default_log_name,
    fp_default_out_data_name,
    fp_task_pattern,
)
from dpgen2.utils.chdir import (
    set_directory,
)
from dpgen2.utils.run_command import (
    run_command,
)
from dpgen2.utils.setup_ele_temp import (
    setup_ele_temp,
)

from ..utils import (
    BinaryFileInput,
)
from .prep_fp import (
    PrepFp,
)
from .run_fp import (
    RunFp,
)

# global static variables
deepmd_input_path = "one_frame_input"

# global static variables
deepmd_temp_path = "one_frame_temp"


class DeepmdInputs:
    @staticmethod
    def args() -> List[Argument]:
        return []

    def __init__(self, **kwargs: Any):
        self.data = kwargs


class PrepDeepmd(PrepFp):
    def prep_task(
        self,
        conf_frame: dpdata.System,
        inputs,
    ):
        r"""Define how one Deepmd task is prepared.

        Parameters
        ----------
        conf_frame : dpdata.System
            One frame of configuration in the dpdata format.
        inputs : str or dict
            This parameter is useless in deepmd.
        """
        conf_frame.to("deepmd/npy", deepmd_input_path)

    @OP.exec_sign_check
    def execute(
        self,
        ip: OPIO,
    ) -> OPIO:
        r"""Prepare one DeepMD task for all configurations.

        All input configurations are gathered into one dpdata.MultiSystems and
        written to a single task directory. The teacher model labels all
        frames in one run.
        """
        confs = ip["confs"]
        type_map = ip["type_map"]

        has_fparam = any(len(list(mm.rglob("fparam.npy"))) > 0 for mm in confs)
        has_aparam = any(len(list(mm.rglob("aparam.npy"))) > 0 for mm in confs)
        if has_fparam:
            setup_ele_temp(False)
        if has_aparam:
            setup_ele_temp(True)

        mixed_type = any(
            len(list(mm.rglob("*/real_atom_types.npy"))) > 0 for mm in confs
        )

        ms = dpdata.MultiSystems(type_map=type_map)
        for mm in confs:
            if mixed_type:
                ms.load_systems_from_file(mm, fmt="deepmd/npy/mixed")
            else:
                ms.from_deepmd_npy(mm, labeled=False)

        task_name = fp_task_pattern % 0
        task_path = Path(task_name)
        with set_directory(task_path):
            if mixed_type:
                ms.to_deepmd_npy_mixed(deepmd_input_path)
            else:
                ms.to_deepmd_npy(deepmd_input_path)

        return OPIO(
            {
                "task_names": [task_name],
                "task_paths": [task_path],
            }
        )


class RunDeepmd(RunFp):
    def input_files(self) -> List[str]:
        r"""The mandatory input files to run a Deepmd task.

        Returns
        -------
        files: List[str]
            A list of madatory input files names.

        """
        return [deepmd_input_path]

    def optional_input_files(self) -> List[str]:
        r"""The optional input files to run a Deepmd task.

        Returns
        -------
        files: List[str]
            A list of optional input files names.

        """
        return []

    def run_task(
        self,
        teacher_model_path: BinaryFileInput,
        out: str,
        log: str,
    ) -> Tuple[str, str]:
        r"""Defines how one FP task runs

        Parameters
        ----------
        command : str
            The command of running Deepmd task
        out : str
            The name of the output data file.

        Returns
        -------
        out_name: str
            The file name of the output data in the dpdata.LabeledSystem format.
        log_name: str
            The file name of the log.
        """
        log_name = log
        out_name = out

        dp, type_map_teacher = self._get_dp_model(teacher_model_path)

        # Run deepmd
        self._dp_infer(dp, type_map_teacher, out_name)

        run_command(f'echo "job finished!" > {log_name}', shell=True)

        return out_name, log_name

    def _get_dp_model(self, teacher_model_path: BinaryFileInput):
        from deepmd.infer import DeepPot  # type: ignore

        ext = os.path.splitext(teacher_model_path.file_name)[-1]
        deepmd_teacher_model = "teacher_model" + ext
        teacher_model_path.save_as_file(deepmd_teacher_model)
        dp = DeepPot(Path(deepmd_teacher_model))

        type_map_teacher = dp.get_type_map()

        os.remove(deepmd_teacher_model)
        return dp, type_map_teacher

    def _dp_infer(self, dp, type_map_teacher, out_name):
        mixed_type = len(
            list(Path(deepmd_input_path).rglob("*/real_atom_types.npy"))
        ) > 0

        if mixed_type:
            ms = dpdata.MultiSystems()
            ms.load_systems_from_file(deepmd_input_path, fmt="deepmd/npy/mixed")
        else:
            ms = dpdata.MultiSystems()
            ms.from_deepmd_npy(deepmd_input_path, labeled=False)

        for ss in ms:
            conf_type_map = ss["atom_names"]
            self._check_type_map(conf_type_map, type_map_teacher)

            atype = np.array(
                [type_map_teacher.index(conf_type_map[j]) for j in ss["atom_types"]]
            )

            nframe = ss.get_nframes()
            coord = ss["coords"]
            cell = None if ss.nopbc else ss["cells"].reshape([nframe, -1])  # type: ignore

            energy, force, virial_force = dp.eval(coord, cell, atype.tolist())

            ss.data["energies"] = energy.reshape([-1])
            ss.data["forces"] = force
            ss.data["virials"] = virial_force.reshape([-1, 3, 3])

        if mixed_type:
            ms.to_deepmd_npy_mixed(out_name)
        else:
            ms.to_deepmd_npy(out_name)

    def _check_type_map(self, conf_type_map, type_map_teacher):
        if not set(conf_type_map).issubset(set(type_map_teacher)):  # type: ignore
            err_message = (
                f"the type map of system ({conf_type_map}) is not subset of "
                + f"the type map of the teacher model ({type_map_teacher})."
            )
            raise FatalError("deepmd labeling failed\n", "err msg", err_message, "\n")

    @staticmethod
    def args() -> List[dargs.Argument]:
        r"""The argument definition of the `run_task` method.

        Returns
        -------
        arguments: List[dargs.Argument]
            List of dargs.Argument defines the arguments of `run_task` method.
        """

        doc_deepmd_teacher_model = (
            "The path of teacher model, which can be loaded by deepmd.infer.DeepPot"
        )
        doc_deepmd_log = "The log file name of dp"
        doc_deepmd_out = "The output dir name of labeled data. In `deepmd/npy` format provided by `dpdata`."
        return [
            Argument(
                "teacher_model_path",
                [str, BinaryFileInput],
                optional=False,
                doc=doc_deepmd_teacher_model,
            ),
            Argument(
                "out",
                str,
                optional=True,
                default=fp_default_out_data_name,
                doc=doc_deepmd_out,
            ),
            Argument(
                "log",
                str,
                optional=True,
                default=fp_default_log_name,
                doc=doc_deepmd_log,
            ),
        ]
