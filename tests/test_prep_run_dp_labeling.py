import os
import shutil
import sys
import unittest
from pathlib import (
    Path,
)

import dpdata
import numpy as np
from dflow.python import (
    FatalError,
)
from mock import (
    Mock,
    patch,
)

from dpgen2.fp.deepmd import (
    PrepDeepmd,
    RunDeepmd,
    deepmd_input_path,
    deepmd_temp_path,
)
from dpgen2.utils import (
    BinaryFileInput,
)


class TestPrepDeepmd(unittest.TestCase):
    def setUp(self):
        self.system = dpdata.System(
            data={
                "atom_names": ["H"],
                "atom_numbs": [1],
                "atom_types": np.zeros(1, dtype=int),
                "cells": np.eye(3).reshape(1, 3, 3),
                "coords": np.zeros((1, 1, 3)),
                "orig": np.zeros(3),
            }
        )
        self.system2 = dpdata.System(
            data={
                "atom_names": ["H"],
                "atom_numbs": [1],
                "atom_types": np.zeros(2, dtype=int),
                "cells": np.eye(3).reshape(1, 3, 3).repeat(2, axis=0),
                "coords": np.zeros((2, 1, 3)),
                "orig": np.zeros(3),
            }
        )

    def tearDown(self):
        if Path(deepmd_input_path).is_dir():
            shutil.rmtree(deepmd_input_path)
        shutil.rmtree("task.000000", ignore_errors=True)
        shutil.rmtree("confs", ignore_errors=True)

    def test_prep_task(self):
        prep_deepmd = PrepDeepmd()
        prep_deepmd.prep_task(
            self.system,
            {},
        )
        ss = dpdata.System(deepmd_input_path, fmt="deepmd/npy")
        self.assertEqual(self.system["atom_names"], ss["atom_names"])
        self.assertEqual(self.system["atom_numbs"], ss["atom_numbs"])
        self.assertTrue(np.allclose(self.system["atom_types"], ss["atom_types"]))
        self.assertTrue(np.allclose(self.system["cells"], ss["cells"]))
        self.assertTrue(np.allclose(self.system["coords"], ss["coords"]))

    def test_execute(self):
        confs_dir = Path("confs")
        confs_dir.mkdir(exist_ok=True)
        ms = dpdata.MultiSystems(type_map=["H"])
        ms.append(self.system)
        ms.append(self.system2)
        ms.to_deepmd_npy(str(confs_dir))

        prep_deepmd = PrepDeepmd()
        op = prep_deepmd.execute(
            {
                "config": {"inputs": {}},
                "type_map": ["H"],
                "confs": [confs_dir],
            }
        )

        self.assertEqual(op["task_names"], ["task.000000"])
        self.assertEqual(len(op["task_paths"]), 1)
        self.assertTrue((op["task_paths"][0] / deepmd_input_path).is_dir())

        out_ms = dpdata.MultiSystems()
        out_ms.from_deepmd_npy(
            op["task_paths"][0] / deepmd_input_path, labeled=False
        )
        self.assertEqual(out_ms.get_nframes(), 3)


class TestRunDeepmd(unittest.TestCase):
    def setUp(self):
        self.system = dpdata.System(
            data={
                "atom_names": ["H", "O"],
                "atom_numbs": [1, 1],
                "atom_types": np.array([0, 1]),
                "cells": np.eye(3).reshape(1, 3, 3),
                "coords": np.zeros((1, 2, 3)),
                "orig": np.zeros(3),
            }
        )
        self.system3frames = dpdata.System(
            data={
                "atom_names": ["H", "O"],
                "atom_numbs": [1, 1],
                "atom_types": np.array([0, 1]),
                "cells": np.eye(3).reshape(1, 3, 3).repeat(3, axis=0),
                "coords": np.zeros((3, 2, 3)),
                "orig": np.zeros(3),
            }
        )
        self.system_nopbc = dpdata.System(
            data={
                "atom_names": ["H", "O"],
                "atom_numbs": [1, 1],
                "atom_types": np.array([0, 1]),
                "cells": np.eye(3).reshape(1, 3, 3),
                "coords": np.zeros((1, 2, 3)),
                "orig": np.zeros(3),
                "nopbc": True,
            }
        )
        self.task_path = Path("task")
        self.task_path.mkdir(parents=True, exist_ok=True)
        Path(self.task_path / "teacher-model.pb").write_bytes(b"0123456789")
        self.teacher_model = BinaryFileInput(self.task_path / "teacher-model.pb")

    def tearDown(self):
        shutil.rmtree(self.task_path, ignore_errors=True)

        shutil.rmtree(deepmd_input_path, ignore_errors=True)

        shutil.rmtree(deepmd_temp_path, ignore_errors=True)

    def test_get_dp_model(self):
        # from deepmd.infer import DeepPot
        deepmd = Mock()
        modules = {"deepmd": deepmd, "deepmd.infer": deepmd.infer}
        deepmd.infer.DeepPot = Mock()
        dp = deepmd.infer.DeepPot.return_value
        with patch.dict("sys.modules", modules):
            run_deepmd = RunDeepmd()
            # test1
            dp.get_type_map.return_value = ["H", "C"]
            _dp, _type_map = run_deepmd._get_dp_model(self.teacher_model)
            self.assertTrue(_dp is dp)
            self.assertEqual(_type_map, ["H", "C"])

    def test_dp_infer(self):
        self.system3frames.to("deepmd/npy", deepmd_input_path)

        out_name = self.task_path / "test_out"

        dp = Mock()
        self._set_mock_dp_eval(dp, self.system3frames)

        run_deepmd = RunDeepmd()
        run_deepmd._dp_infer(dp, ["H", "O"], str(out_name))

        self.assertTrue(out_name.is_dir())
        self._check_output_system(out_name, self.system3frames)

    def test_run_task(self):
        self.system3frames.to("deepmd/npy", deepmd_input_path)

        out_name = self.task_path / "test_out"
        log_name = self.task_path / "test.log"

        run_deepmd = RunDeepmd()

        # run run_task
        deepmd = Mock()
        modules = {"deepmd": deepmd, "deepmd.infer": deepmd.infer}
        deepmd.infer.DeepPot = Mock()
        dp = deepmd.infer.DeepPot.return_value
        self._set_mock_dp_eval(dp, self.system3frames)
        with patch.dict("sys.modules", modules):
            run_deepmd.run_task(
                self.teacher_model,
                str(out_name),
                str(log_name),
            )
            self.assertTrue(log_name.is_file())
            self.assertTrue(out_name.is_dir())
            self._check_output_system(out_name, self.system3frames)

    def _set_mock_dp_eval(self, dp, system):
        energy, force, virial_force = self._get_labels(system)
        dp.eval.return_value = (energy, force, virial_force)
        dp.get_type_map.return_value = ["O", "H"]

    def _check_output_system(self, path, system):
        ss = dpdata.LabeledSystem(path, fmt="deepmd/npy")
        energy, force, virial_force = self._get_labels(system)

        self.assertTrue(ss["atom_numbs"] == system["atom_numbs"])
        self.assertTrue(np.allclose(ss["atom_types"], system["atom_types"]))
        self.assertTrue(np.allclose(ss["energies"], energy))
        self.assertTrue(np.allclose(ss["forces"], force))
        self.assertTrue(np.allclose(ss["virials"], virial_force))

    def _get_labels(self, system):
        energy = np.zeros(system["coords"].shape[0])
        force = np.zeros_like(system["coords"])
        virial_force = np.zeros((system["coords"].shape[0], 3, 3))
        return energy, force, virial_force

    def test_dp_infer_with_nopbc(self):
        self.system_nopbc.to("deepmd/npy", deepmd_input_path)

        out_name = self.task_path / "test_out"

        dp = Mock()
        self._set_mock_dp_eval(dp, self.system_nopbc)

        run_deepmd = RunDeepmd()
        run_deepmd._dp_infer(dp, ["H", "O"], str(out_name))

        dp.eval.assert_called_once()
        self.assertTrue(
            np.allclose(dp.eval.call_args[0][0], self.system_nopbc["coords"])
        )
        self.assertIsNone(dp.eval.call_args[0][1])
        self.assertEqual(
            dp.eval.call_args[0][2], [1, 0]
        )
