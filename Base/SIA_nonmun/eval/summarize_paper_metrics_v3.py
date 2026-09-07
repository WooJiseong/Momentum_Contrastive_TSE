import json
import sys
import numpy as np
from pathlib import Path

p = Path(sys.argv[1])
ver = sys.argv[2] if len(sys.argv) > 2 else p.stem

x = json.load(open(p))

rows = x["results"]["1"]

def arr(k):
    return np.asarray(
        [
            float(r.get(k, np.nan))
            for r in rows
        ],
        dtype=float,
    )

def stat(k):
    a = arr(k)
    a = a[np.isfinite(a)]

    if not len(a):
        return np.nan, np.nan

    return a.mean(), a.std()

print()
print("=" * 82)
print(f"{ver} — PHYSICAL-SCALE TEST")
print("=" * 82)
print("N =", len(rows))
print(
    "output rescale = inference m "
    "(predicted mode: deployable m_pred)"
)

print()
print("OVERALL")
print("-" * 82)

keys = [
    "snr",
    "input_snr",
    "snri",
    "si_snr",
    "input_si_snr",
    "si_snri",
    "si_sdr",
    "in_si",
    "si_sdri",
    "pesq",
    "stoi",
    "output_scale_m",
    "true_m",
]

for k in keys:
    mu, sd = stat(k)

    if np.isfinite(mu):
        print(
            f"{k:24s}"
            f"{mu:10.4f} ± {sd:9.4f}"
        )

sdri = arr("si_sdri")
valid = np.isfinite(sdri)

fail = (
    int(np.sum(sdri[valid] < 0))
    if valid.any()
    else 0
)

print(
    f"SI-SDRi<0 failure"
    f"{fail:10d}/{valid.sum()} "
    f"({100*fail/max(1,valid.sum()):.2f}%)"
)


# ------------------------------------------------------------
# SNR bins — 전체 test distribution을 덮도록 구성
# ------------------------------------------------------------

snr = arr("input_snr")

bins = [
    (-np.inf, -12),
    (-12, -9),
    (-9, -6),
    (-6, -3),
    (-3, 0),
    (0, 3),
    (3, 6),
    (6, np.inf),
]

print()
print("INPUT-SNR BIN")
print("-" * 82)

print(
    f"{'SNR':>14}"
    f"{'N':>7}"
    f"{'SNRi':>10}"
    f"{'SI-SNRi':>11}"
    f"{'SI-SDRi':>11}"
    f"{'PESQ':>9}"
    f"{'STOI':>9}"
    f"{'FAIL%':>9}"
)

covered = 0

for lo, hi in bins:

    mask = np.isfinite(snr)

    if np.isfinite(lo):
        mask &= snr >= lo

    if np.isfinite(hi):
        mask &= snr < hi

    n = int(mask.sum())

    if not n:
        continue

    covered += n

    def mean_mask(k):
        a = arr(k)[mask]
        a = a[np.isfinite(a)]
        return np.mean(a) if len(a) else np.nan

    d = arr("si_sdri")[mask]
    d = d[np.isfinite(d)]

    fail_pct = (
        100 * np.mean(d < 0)
        if len(d)
        else np.nan
    )

    if not np.isfinite(lo):
        label = f"< {hi:g}"
    elif not np.isfinite(hi):
        label = f">= {lo:g}"
    else:
        label = f"[{lo:g},{hi:g})"

    print(
        f"{label:>14}"
        f"{n:7d}"
        f"{mean_mask('snri'):10.4f}"
        f"{mean_mask('si_snri'):11.4f}"
        f"{mean_mask('si_sdri'):11.4f}"
        f"{mean_mask('pesq'):9.4f}"
        f"{mean_mask('stoi'):9.4f}"
        f"{fail_pct:9.2f}"
    )

print()
print(
    f"SNR bins covered = "
    f"{covered}/{len(rows)}"
)

print("=" * 82)
