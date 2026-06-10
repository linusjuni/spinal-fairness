# 08 — Carbon Footprint of Training

> **Status (2026-06-10):** GPU-hours measured from LSF logs; emissions estimated with the
> standard ML-CO₂ methodology. A100 board confirmed as **A100-PCIE-40GB (250 W)** via
> `nvidia-smi -L`. Central estimate **≈ 53–74 kg CO₂e** (~534 kWh) for the full training
> workload (**30 completed model folds** across Dataset001/002/003 — now includes the two
> retrained Dataset003 silver `3d_fullres` folds 1 and 4).

## Summary

| Quantity | Value |
|---|---|
| Total GPU-hours | 1,105 h (486.3 A100 + 618.9 L40S) |
| Energy (central) | ≈ 534 kWh |
| **Emissions (central)** | **52.9 kg CO₂e** (observed grid) – **73.7 kg CO₂e** (adjusted grid) |
| Sensitivity range | 52–78 kg CO₂e |
| Relatable equivalent | ≈ 500–690 km by an average EU car (106.4 gCO₂/km) |

L40S dominates the footprint (~64% of energy): it ran more hours at higher board power
than the A100. Denmark's low-carbon, wind-heavy grid makes this ~3–5× lower than the same
compute would emit on the world-average grid (~475 gCO₂e/kWh).

> **✅ Updated (2026-06-10):** the two Dataset003 silver `3d_fullres` folds finished and are
> now included. Actual LSF run times: fold 1 (job 28602550) **35.53 h**, fold 4 (job 28602611)
> **33.40 h** — **+68.9 A100-h** (slightly above the earlier 62-h estimate; `n_proc_DA=1`
> retrains run a touch slower than the original silver 3d folds at 28.8–32.2 h). New totals:
> **1,105 GPU-h, ~534 kWh, ≈ 53–74 kg CO₂e** (central PUE, observed→adjusted grid).

## Methodology

Energy and emissions follow the standard ML-CO₂ accounting (Lacoste et al. 2019; Strubell
et al. 2019; Patterson et al. 2021):

```
E_kWh   = Σ_GPU ( GPU-hours × TDP_kW × PUE )
CO₂e    = E_kWh × CI_grid
```

Per the MLCO₂ convention, GPUs are charged at **100% TDP** because per-job GPU utilization
was not logged. This is a deliberately conservative (slight over-) estimate; true emissions
are likely at or below the central figure. The estimate covers **GPU energy with datacenter
PUE overhead** — host CPU/DRAM/networking are excluded and would add ~10–30%.

## Inputs and Assumptions

| Parameter | Value | Source |
|---|---|---|
| A100 GPU-hours | 486.3 h (incl. silver 3d folds 1+4: 35.53+33.40 h) | LSF logs (`jobs/logs/`, this work) |
| L40S GPU-hours | 618.9 h | LSF logs (`jobs/logs/`, this work) |
| A100 board power | 250 W (PCIe-40GB, confirmed via `nvidia-smi -L`) | NVIDIA A100 datasheet |
| L40S board power | 350 W | NVIDIA L40S datasheet |
| PUE | 1.58 central; 1.55–1.67 range | Strubell et al. 2019; Green Algorithms |
| Danish grid intensity (2023) | 99 g/kWh (observed) – 138 g/kWh (adjusted) | Danish Energy Agency |
| Car-km factor | 106.4 gCO₂/km (EU new car, WLTP); 249 gCO₂/km (US fleet) | EEA; US EPA |

GPU-hour split by queue (see `03_training.md`, `06_gold_silver_training.md`):
A100 = Dataset001 `2d` (5 folds) + Dataset003 silver all 10 folds; L40S = Dataset001
`3d_fullres` (5 folds) + Dataset002 gold all 10 folds.

## Scenario Table

A100 fixed at 250 W (confirmed); only PUE and grid basis vary.

| Scenario | Total kWh | @ 99 g/kWh | @ 138 g/kWh |
|---|---|---|---|
| Low (PUE 1.55) | 524 | 51.9 kg | 72.3 kg |
| **Central (PUE 1.58)** | **534** | **52.9 kg** | **73.7 kg** |
| High (PUE 1.67) | 565 | 55.9 kg | 77.9 kg |

Component split (central): A100 192 kWh (19.0–26.5 kg), L40S 342 kWh (33.9–47.2 kg).

## Suggested Paper Wording

> Training the segmentation models consumed 1,105 GPU-hours (486 A100-h + 619 L40S-h).
> Following the ML CO₂ methodology (Lacoste et al. 2019; Strubell et al. 2019; Patterson
> et al. 2021), assuming 100% TDP (A100-PCIE-40GB 250 W, L40S 350 W) and a PUE of 1.58, this
> corresponds to ≈534 kWh. At the 2023 Danish grid intensity (99–138 gCO₂e/kWh; Danish
> Energy Agency), the estimated footprint is 53–74 kg CO₂e (≈500–690 km by an average car).
> Denmark's low-carbon, wind-heavy grid makes this 3–5× lower than the same compute on the
> world-average grid.

## Caveats

1. **A100 board confirmed** as A100-PCIE-40GB (250 W) via `nvidia-smi -L` on a `gpua100` node —
   this was previously the largest uncertainty and is now resolved. Remaining variation is
   only PUE and grid basis, so the range is tight (52–78 kg).
2. **GPU-only** estimate — host CPU/DRAM/networking excluded (~10–30% more for an exclusive
   single-GPU job).
3. **Grid basis** — Denmark reports observed (99) and adjusted (138 g/kWh) intensities for
   2023; pick one, cite the year, and acknowledge the range. Figures are time-sensitive.
4. **No DTU-specific PUE** was available; the 1.55–1.67 global-average range is a substitute,
   not a measured site value.

## Citations (BibTeX)

```bibtex
@article{lacoste2019quantifying,
  title   = {Quantifying the Carbon Emissions of Machine Learning},
  author  = {Lacoste, Alexandre and Luccioni, Alexandra and Schmidt, Victor and Dandres, Thomas},
  journal = {arXiv preprint arXiv:1910.09700},
  year    = {2019}
}

@inproceedings{strubell2019energy,
  title     = {Energy and Policy Considerations for Deep Learning in {NLP}},
  author    = {Strubell, Emma and Ganesh, Ananya and McCallum, Andrew},
  booktitle = {Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL)},
  pages     = {3645--3650},
  year      = {2019}
}

@article{patterson2021carbon,
  title   = {Carbon Emissions and Large Neural Network Training},
  author  = {Patterson, David and Gonzalez, Joseph and Le, Quoc and Liang, Chen and
             Munguia, Lluis-Miquel and Rothchild, Daniel and So, David and Texier, Maud and Dean, Jeff},
  journal = {arXiv preprint arXiv:2104.10350},
  year    = {2021}
}

@misc{nvidia_a100_datasheet,
  title        = {{NVIDIA A100 Tensor Core GPU} Datasheet},
  author       = {{NVIDIA Corporation}},
  howpublished = {\url{https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-nvidia-us-2188504-web.pdf}},
  note         = {Max TDP: PCIe 40GB 250 W, PCIe 80GB 300 W, SXM 400 W}
}

@misc{nvidia_l40s,
  title        = {{NVIDIA L40S} GPU for AI and Graphics Performance},
  author       = {{NVIDIA Corporation}},
  howpublished = {\url{https://www.nvidia.com/en-us/data-center/l40s/}},
  note         = {Max power consumption: 350 W}
}

@misc{dea_keyfigures2023,
  title        = {Key Figures: {CO2} Emissions per kWh Electricity Sold},
  author       = {{Danish Energy Agency}},
  year         = {2023},
  howpublished = {\url{https://ens.dk/en/analyses-and-statistics/key-figures}},
  note         = {Observed 99 g/kWh; adjusted 138 g/kWh}
}

@misc{eea_newcars2023,
  title        = {{CO2} Performance of New Passenger Cars in Europe},
  author       = {{European Environment Agency}},
  year         = {2024},
  howpublished = {\url{https://www.eea.europa.eu/en/analysis/indicators/co2-performance-of-new-passenger}},
  note         = {Average new car 2023: 106.4 gCO2/km (WLTP)}
}

@misc{epa_typicalvehicle,
  title        = {Greenhouse Gas Emissions from a Typical Passenger Vehicle},
  author       = {{U.S. Environmental Protection Agency}},
  howpublished = {\url{https://www.epa.gov/greenvehicles/greenhouse-gas-emissions-typical-passenger-vehicle}},
  note         = {~400 g CO2/mile (~249 g/km), fleet tailpipe}
}
```
