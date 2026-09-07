"""Optional full PN baseline separator integration.

The public repository contains the PN encode path used by training, but not the
full proposed-monaural separator implementation required only for TensorBoard
baseline audio. Keep baseline.log_audio=false unless that implementation is
provided locally.
"""


def load_pn_full(device):
    raise RuntimeError(
        'Full PN baseline separator is not bundled with the public repository. '
        'Set baseline.log_audio: false (default) or provide a compatible '
        'baseline_pnenroll.load_pn_full implementation locally.'
    )
