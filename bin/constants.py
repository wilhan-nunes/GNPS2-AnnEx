# -*- coding: utf-8 -*-
import os

SUPPORTED_WORKFLOWS = {"auto", "classic", "fbmn", "librarysearch"}
ENTROPY_USI_TIMEOUT_SEC = int(os.environ.get("ENTROPY_USI_TIMEOUT_SEC", 8))
LIBRARY_RESULTS_CACHE_TIMEOUT_SEC = 86400

# SharedPeaks normalisation: x / (x + K) maps raw peak counts to [0, 1].
K_SHARED_PEAKS = 5

# MZ precision factor.
# Gate mode (PPM_GATE_MODE=True): 1.0 within tolerance, PPM_FLOOR beyond (step function).
# Gaussian mode (PPM_GATE_MODE=False): f + (1−f)·exp(−max(0, |ppm|−t)² / (2·S²))
PPM_TOLERANCE = 5.0    # t — tolerance threshold (ppm), fixed
PPM_SIGMA = 10.0       # S — Gaussian decay width; only used when PPM_GATE_MODE=False
PPM_FLOOR = 0.7        # f — penalty floor (applied beyond tolerance in gate mode)
PPM_GATE_MODE = True   # True = step gate; False = Gaussian decay

# Bayesian shrinkage pseudo-counts for support_fraction.
# Shrinks naive support/total toward the prior rate (α/(α+β)=0.5 with α=β=1),
# preventing trivial 1.0 scores for low-evidence single-hit scans.
SUPPORT_ALPHA = 1.0
SUPPORT_BETA = 1.0

# Library diversity soft-cap: min(unique_source_libraries, K) / K → [0, 1].
K_LIBRARY_DIVERSITY = 5

# All-unknown-structure penalty.
# Applied when every hit in a scan lacks BOTH a SMILES and an InChIKey, i.e. the
# scan has no structural evidence at all. Without a structure we cannot offer a
# confident annotation, so the score is demoted regardless of spectral quality.
#   penalized = min(score × UNKNOWN_STRUCTURE_PENALTY, UNKNOWN_STRUCTURE_CAP)
# UNKNOWN_STRUCTURE_PENALTY = 1.0 disables the multiplier; CAP = 100.0 disables the cap.
UNKNOWN_STRUCTURE_PENALTY = 0.5   # multiplicative demotion (preserves spectral ranking)
UNKNOWN_STRUCTURE_CAP = 55.0      # hard ceiling: an all-unknown scan can never read green
