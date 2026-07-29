"""Central configuration for the PPG-BreakHis experiments.

Every hyperparameter and ablation switch lives here so the rest of the code
stays clean. Override any field from the command line in train.py.
"""
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Config:
    # ---- data ----
    data_root: str = "./BreakHis_v1"          # folder that contains the images
    magnification: str = "200X"                # 40X | 100X | 200X | 400X
    image_size: int = 224
    val_frac: float = 0.10                      # fractions are PATIENT-level, not image-level
    test_frac: float = 0.20
    num_classes: int = 2                        # benign vs malignant
    class_names: tuple = ("benign", "malignant")

    # ---- backbone ----
    backbone: str = "swin_tiny_patch4_window7_224"
    pretrained: bool = True
    feature_stage: int = 3                      # 2 -> 1/16 (384ch), 3 -> 1/32 (768ch, matches D=768)

    # ---- prototypes ----
    protos_per_class: int = 10                  # K in the write-up; total P = K * num_classes
    temperature: float = 0.1                    # tau for the spatial softmax

    # ---- uncertainty ----
    mc_samples: int = 5                         # M (set to 1 to disable uncertainty)
    mc_dropout_p: float = 0.2                   # dropout applied to the feature map per MC pass

    # ---- gating (ablation switches) ----
    gate_direction: Literal["corrected", "original"] = "corrected"
    use_bernoulli_gate: bool = True             # the z gate
    use_uncertainty_gate: bool = True           # the g (HGLU-style) gate
    pooling: Literal["attention", "maxpool"] = "attention"
    head: Literal["fixed", "learnable"] = "fixed"   # fixed is the STABLE, interpretable
                                                    # default; empirically the learnable head
                                                    # collapses to the majority class on some
                                                    # seeds under weak signal. Kept as ablation.
    beta: float = 5.0                           # gate steepness
    gamma: float = 0.1                          # gate threshold on uncertainty
    gumbel_tau: float = 1.0                     # STARTING relaxation temperature for z
    gumbel_tau_min: float = 0.3                 # annealed DOWN to this over training
    gate_bias_init: float = 3.0                 # larger -> gates start more open (stabler)

    # ---- training ----
    epochs: int = 30
    batch_size: int = 16
    lr: float = 1e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 2
    class_balanced_loss: bool = True            # weight CE by inverse class frequency
    num_workers: int = 4
    seed: int = 0
    device: str = "cuda"

    # ---- logging ----
    out_dir: str = "./runs"
    exp_name: str = "ppg_swint"

    def total_protos(self) -> int:
        return self.protos_per_class * self.num_classes
