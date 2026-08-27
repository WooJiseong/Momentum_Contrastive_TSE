# STFT Magnitude-Overlap Gate

## 목적

`target_orthogonal_leakage_loss`의 기존 waveform projection Gate는

```text
||nuisance - proj_target(nuisance)||^2 / (||nuisance||^2 + eps)
```

를 사용한다. 서로 다른 음성 waveform은 시간축의 내적이 작아 이 값이 거의
항상 1에 가까워질 수 있다. 이 경우 `orthogonal_gate_threshold`가 실질적으로
window를 걸러내지 못한다.

새 `stft_magnitude_overlap` 모드는 각 waveform window에 STFT를 적용한 뒤,
target과 개별 nuisance source의 magnitude 스펙트럼을 비교한다.

```text
M_t = abs(STFT(target_window))
M_n = abs(STFT(nuisance_window))
overlap = <M_t, M_n> / (||M_t||_2 ||M_n||_2 + eps)
gate = 1[overlap <= stft_magnitude_overlap_threshold]
```

즉, 현재 구현은 target과 spectral magnitude가 충분히 겹치지 않는
nuisance window를 leakage 계산 대상으로 선택한다. 이후 target/nuisance
activity Gate가 추가로 적용된다. STFT는 Gate 계산에만 사용하고, 실제
leakage 값은 기존처럼 waveform에서 계산한다.

## YAML 설정

기존 설정에 다음 항목을 추가하면 네 Stage1 Lab
(`PN_Indiv_MOCOCO`, `Soft_MOCOCO`, `Soft_Indiv_MOCOCO`, `Attn_MOCOCO`)에서
동일하게 사용할 수 있다.

```yaml
loss:
  type: si_sdr_plus_target_orthogonal_leakage
  orthogonal_leakage_weight: 0.1
  orthogonal_window_size: 2048
  orthogonal_hop_size: 1024
  gate_mode: stft_magnitude_overlap
  stft_n_fft: 510
  stft_hop_length: 128
  stft_win_length: 510
  stft_center: false
  stft_magnitude_overlap_threshold: 0.85
  normalize_residual_energy: true
  activity_gate: true
  target_activity_threshold: 0.01
  nuisance_activity_threshold: 0.01
```

`0.85`는 시작점일 뿐이다. 첫 validation epoch의
`val_stft_magnitude_overlap`와 `val_stft_magnitude_gate_ratio`를 확인한다.
gate ratio가 0에 가깝다면 threshold를 높이고, 거의 1이면 낮춘다. 목표는
activity Gate 적용 전 `stft_magnitude_gate_ratio`가 모든 window를 통과하거나
모두 제거하지 않는 구간이다.

기존 YAML처럼 `gate_mode`를 생략하면 `waveform_orthogonal`이 사용되므로
기존 실험의 결과와 재현성은 유지된다.

## 로깅 지표

- `*_stft_magnitude_overlap`: batch에서 계산된 overlap 평균
- `*_stft_magnitude_gate_ratio`: activity Gate 적용 전 STFT Gate 통과 비율
- `*_orthogonal_gate_ratio`: activity Gate까지 적용한 최종 통과 비율

## 실행 코드 변경 범위

별도의 Loss 파일을 만들어 import 경로를 바꿀 필요가 없다. 공통
`Base/Code_Snippet/loss_code.py`에 backward-compatible한 모드와 설정 변환
helper를 추가했고, 네 Stage1 실행부는 `loss` 설정을 helper로 전달한다.
새 모드를 쓰는 경우 실행 Python 파일을 다시 뜯어고치지 않고 YAML만 바꾸면
된다. 별도 Loss 함수를 추가해 함수명까지 바꾸는 경우에만 각 Lab의 local
`losses.py`와 호출부를 수정해야 한다.
