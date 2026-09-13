"""Pin the manuscript device before pytest imports test modules."""
import os

os.environ["FUSION_DEVICE"] = "arc_16pancake_nuc600"
