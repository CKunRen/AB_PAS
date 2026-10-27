# AB-PAS: adaptive boundary sampling with parsimonious polynomial active subspaces

Code for the manuscript *Adaptive boundary sampling with parsimonious polynomial
active subspaces for high-dimensional multi-output reliability analysis*.

AB-PAS separates label acquisition from response-model selection under one
charged budget of true-model evaluations: an orthogonal encoder with a
multi-output neural response map allocates new labels between component
boundaries and global exploration, and a ladder of polynomial experts fitted to
the same non-holdout labels is screened on a fixed holdout, with a neural
fallback when no candidate passes. No gradients of the true model are used.

## What this repository contains

Only the code of the method and of the mathematical benchmark suite:

```
event_adaptive.py               sampling layer (methods: static, active8, conflict, fixed2, full)
event_adaptive_v3.py            sampling layer used by AB-PAS
event_adaptive_counterexample.py  auxiliary driver used for the budget ladder
poly_experts.py                 polynomial experts and the one-standard-error rule
method_benchmarks.py            seven mathematical benchmark families
reproducibility.py              runtime metadata helpers
run_v6_final.py                 mathematical experiments (Table 4, Figures 2-5)
baselines_v2.py                 AK-MCS, AS-GP and AS-AK baselines
make_math_main_hatched.py       Figure 2
make_estimate_figure.py         Figure 3
make_c5_boundary.py             Figure 7
```

## What this repository does not contain

By design, the release does not include:

* the semi-analytical transmission module used for the engineering case study
  (`gear_module.py`, its configuration and the 141-input specification);
* the frozen result files behind Tables 4 and 9 and the associated figures;
* the figure images themselves.

These are available from the corresponding author on reasonable request.

## Environment

Python 3.12 with numpy 1.26, scipy 1.15, scikit-learn 1.6, torch 2.11 and
matplotlib 3.10 (see `requirements.txt`). Everything runs on CPU; the drivers
set `torch.set_num_threads(2)`. On Windows, if importing torch and numpy in the
same process raises an OpenMP initialisation error, set
`KMP_DUPLICATE_LIB_OK=TRUE` before running.

## Reproducing the mathematical results

```
python run_v6_final.py --seeds 10     # AB-PAS on the seven families
python baselines_v2.py                # AK-MCS, AS-GP, AS-AK
python make_math_main_hatched.py      # Figure 2
python make_estimate_figure.py        # Figure 3
python make_c5_boundary.py            # Figure 7 (about 6 min)
```

Results are written as JSON under `results/` and figures under `figures/`;
both directories are created by the scripts. Trajectory seeds are
`410000 + 1000*j + s` for mathematical case *j*, with `s = 0..9`.

## License

MIT.
