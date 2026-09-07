# DSnOS_MOCO_Stage1

DSnOS Stage0의 동일한 Best encoder를 고정하여 `train-clean-360`만으로 Stage1 세 가지를 비교한다.

| 실험 | Decoder/Loss | GPU |
|---|---|---:|
| `tfgridnet_default` | 기본 PN-MOCOCO TFGridNet, SI-SDR 학습 | 2 |
| `flow_mrjitter_gamma0` | SIA Flow, `gamma=0`, MR-jitter `sigma=0.15` | 2 |
| `flow_dualdso_gamma0` | 위 Flow 설정 + `Base/Code_Snippet/dualdso_loss.py` | 2 |

모든 실험은 다음 encoder를 사용한다.

```text
exp/20260906_dsnos_moco_stage1/dsnos_best_epoch208_pn_encoder.pt
```

`speedups.py`는 각 Python entrypoint에서 PN/ESPnet import 전에 로드된다. 세 로그는 하나의 TensorBoard에 각각 `tfgridnet_default`, `flow_mrjitter_gamma0`, `flow_dualdso_gamma0` 이름으로 연결한다.
