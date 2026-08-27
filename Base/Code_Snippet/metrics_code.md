# 공통 Waveform Metric 기준

모든 Lab의 Stage1 validation/evaluation은
`Base/TSE-through-Positive-Negative-Enroll/eval_monaural.py`와 동일한 규칙을
사용한다.

1. 추정 음성과 `-추정 음성`을 각각 target과 비교한다.
2. MSE가 더 작은 polarity를 선택한다. 별도의 amplitude rescaling은 하지 않는다.
3. 다음 지표를 waveform 전체 `[B, T]` 단위로 계산한다.
   - `si_sdr`: 기존 결과 호환을 위한 `scale_invariant_signal_distortion_ratio`
   - `sdr`: Base와 동일한 `signal_distortion_ratio(zero_mean=True, load_diag=False)`
   - `si_snr`: `scale_invariant_signal_noise_ratio`
   - `snr`: `signal_noise_ratio`
4. `si_sdri`, `sdri`, `si_snri`, `snri`는 동일한 mixture 입력 지표를 빼서 계산한다.

기존 `si_sdr`/`snr` 결과의 이름과 의미는 유지하고, Base 기준의 `sdr`/`si_snr`
계열을 추가했다. 따라서 과거 결과와의 비교가 끊기지 않으면서 Base 구현과의
수치 비교도 가능하다.
