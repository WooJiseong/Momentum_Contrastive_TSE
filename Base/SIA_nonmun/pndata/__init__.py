# pndata: import-isolated copy of OUR online LibriSpeech+WHAM mixer (harvested verbatim from
# sia_fm_tse/NoisyFlowTSE/dataset.py). Kept separate from MeanFlow's own `data/` package so the
# reference tree stays byte-identical; the only bridge is data/datasets.py, which imports the class
# below and adapts its 3-tuple output into MeanFlow's batch dict.
from .online_mixer import NoisyFlowTSEDataset

__all__ = ["NoisyFlowTSEDataset"]
