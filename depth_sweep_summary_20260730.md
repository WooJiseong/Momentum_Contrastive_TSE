# 20260730 TFGridNet Depth Sweep Summary

Fixed settings: `num_epochs=50`, `fusion_layer=range(n_layers)`, 5000-item eval after train.

## Validation Best

| Depth | Condition | n_val | Best val SI-SDRi | Best val SI-SDR | Best step | Eval SI-SDRi | Eval SI-SDR | Eval SNRi |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 3 | baseline | 50 | 8.278073 | 1.866089 | 14397 | 8.114721 | 2.702478 | 10.001622 |
| 3 | mococo | 50 | 8.286964 | 1.874980 | 15336 | 8.098542 | 2.686299 | 9.718286 |
| 4 | baseline | 50 | 8.308477 | 1.896493 | 14710 | 8.148591 | 2.736348 | 8.953554 |
| 4 | mococo | 50 | 8.311275 | 1.899290 | 15336 | 8.127020 | 2.714777 | 8.352696 |
| 6 | baseline | 50 | 8.321670 | 1.909685 | 14397 | 8.152675 | 2.740432 | 6.524708 |
| 6 | mococo | 50 | 8.317135 | 1.905150 | 15336 | 8.131094 | 2.718851 | 5.182864 |

## MoCo Minus Baseline

| Depth | Eval SI-SDRi delta | Eval SI-SDR delta | Best val SI-SDRi delta |
|---:|---:|---:|---:|
| 3 | -0.016179 | -0.016179 | +0.008891 |
| 4 | -0.021571 | -0.021571 | +0.002798 |
| 6 | -0.021581 | -0.021581 | -0.004535 |
