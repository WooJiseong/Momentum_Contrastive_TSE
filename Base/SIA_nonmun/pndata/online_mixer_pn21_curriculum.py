from __future__ import annotations

import multiprocessing as mp

from pndata.online_mixer_pn7 import (
    NoisyFlowTSEDataset as _PN7NoisyFlowTSEDataset,
)


class NoisyFlowTSEDataset(_PN7NoisyFlowTSEDataset):

    def __init__(self, *args, **kwargs):
        initial_prob = float(
            kwargs.get("hard_negative_prob", 0.0)
        )

        self._hard_negative_prob_shared = mp.Value(
            "d",
            initial_prob,
            lock=True,
        )

        super().__init__(*args, **kwargs)

    @property
    def hard_negative_prob(self):
        with self._hard_negative_prob_shared.get_lock():
            return float(
                self._hard_negative_prob_shared.value
            )

    @hard_negative_prob.setter
    def hard_negative_prob(self, value):
        value = float(value)

        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"hard_negative_prob must be in [0,1], got {value}"
            )

        with self._hard_negative_prob_shared.get_lock():
            self._hard_negative_prob_shared.value = value

    def set_hard_negative_prob(self, value):
        self.hard_negative_prob = value
