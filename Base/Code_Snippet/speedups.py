"""Two drop-in speedups for the PN-Enroll stack. Import this BEFORE anything else.

    import speedups   # noqa: F401  -- must come before resemblyzer / espnet imports

Both replace a hot spot inside an installed library with an equivalent that produces byte-identical
output. Neither changes any model, any hyper-parameter, or any data. Nothing here needs the library
to be edited on disk, so it survives a reinstall and travels with the repository.

  1. espnet TFGridNet's F.unfold
     TFGridNet calls F.unfold on a [N, C, L, 1] tensor with a (k, 1) kernel -- a plain sliding window
     along L. PyTorch routes that to im2col, which launches one CUDA kernel PER BATCH ELEMENT (about
     39,000 launches for one encode at batch 8). Tensor.unfold is the same thing as a view, in one
     kernel. Measured: encode 1109 ms -> 762 ms, torch.equal(before, after) == True.

  2. resemblyzer's trim_long_silences
     It builds the PCM buffer as struct.pack("%dh" % n, *array), which explodes a 190k-element numpy
     array into Python varargs and costs more than the VAD it feeds. numpy writes the same bytes
     directly. Measured: 17.0 ms -> 2.6 ms per file, output arrays identical on every file tested.

Verify on your own machine with:  python speedups.py
"""


def _install_fast_unfold():
    import espnet2.enh.separator.tfgridnet_separator as tfgridnet

    class _FunctionalWithFastUnfold:
        def __init__(self, base):
            self._base = base

        def __getattr__(self, name):
            return getattr(self._base, name)

        def unfold(self, inp, kernel_size, dilation=1, padding=0, stride=1):
            k_h, k_w = kernel_size if isinstance(kernel_size, (tuple, list)) else (kernel_size,) * 2
            s_h, s_w = stride if isinstance(stride, (tuple, list)) else (stride,) * 2
            if (inp.dim() == 4 and inp.shape[-1] == 1 and k_w == 1 and s_w == 1
                    and padding == 0 and dilation == 1):
                x = inp[..., 0]                                  # [N, C, L]
                windows = x.unfold(2, k_h, s_h)                  # [N, C, L', k_h]
                return windows.permute(0, 1, 3, 2).reshape(x.shape[0], x.shape[1] * k_h, -1)
            return self._base.unfold(inp, kernel_size, dilation, padding, stride)

    if not isinstance(tfgridnet.F, _FunctionalWithFastUnfold):
        tfgridnet.F = _FunctionalWithFastUnfold(tfgridnet.F)


def _install_fast_trim():
    import resemblyzer
    import resemblyzer.audio as rz

    def trim_long_silences(wav):
        """resemblyzer.trim_long_silences with the float-to-PCM step done in numpy.

        Window size, VAD mode, moving average and dilation are all read from the library at call
        time, so a version bump cannot leave this silently out of step."""
        samples_per_window = (rz.vad_window_length * rz.sampling_rate) // 1000
        wav = wav[:len(wav) - (len(wav) % samples_per_window)]

        pcm_wave = rz.np.round(wav * rz.int16_max).astype("<i2").tobytes()

        voice_flags = []
        vad = rz.webrtcvad.Vad(mode=3)
        for window_start in range(0, len(wav), samples_per_window):
            window_end = window_start + samples_per_window
            voice_flags.append(vad.is_speech(pcm_wave[window_start * 2:window_end * 2],
                                             sample_rate=rz.sampling_rate))
        voice_flags = rz.np.array(voice_flags)

        def moving_average(array, width):
            padded = rz.np.concatenate(
                (rz.np.zeros((width - 1) // 2), array, rz.np.zeros(width // 2)))
            ret = rz.np.cumsum(padded, dtype=float)
            ret[width:] = ret[width:] - ret[:-width]
            return ret[width - 1:] / width

        audio_mask = moving_average(voice_flags, rz.vad_moving_average_width)
        audio_mask = rz.np.round(audio_mask).astype(bool)
        audio_mask = rz.binary_dilation(audio_mask, rz.np.ones(rz.vad_max_silence_length + 1))
        audio_mask = rz.np.repeat(audio_mask, samples_per_window)
        return wav[audio_mask == True]

    # Both names: modules do `from resemblyzer import trim_long_silences`, which reads the package
    # attribute, and anything already holding a reference keeps the old one -- hence "import first".
    rz.trim_long_silences = trim_long_silences
    resemblyzer.trim_long_silences = trim_long_silences


_install_fast_unfold()
_install_fast_trim()
