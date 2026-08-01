"""Central configuration for the PPG sign-language experiments.

Every hyperparameter and ablation switch lives here so the rest of the code
stays clean. Override any field from the command line in train.py.

Ported from BreakHis histopathology to the Arabic Sign Language "Mosl_alphabet"
alphabet (32 classes). Data is two physically separate folders (train / test).
"""
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Config:
    # ---- data (Arabic Sign Language "Mosl_alphabet", 32 classes) ----
    # Two physically separate folders under data_root -> no train/test leakage.
    data_root: str = "/kaggle/input/datasets/youssefnouiouar1/sing-language-recognition/SLR"
    train_subdir: str = "Mosl_alphabet_train"   # -> stratified 80/20 train/val
    test_subdir: str = "Mosl_alphabet_test"     # -> held-out test set
    image_size: int = 224
    val_frac: float = 0.20                       # 80/20 train/val split of the train folder
    num_classes: int = 32
    class_names: tuple = (
        "ain", "al", "aleff", "bb", "dal", "dha", "dhad", "fa", "gaaf", "ghain",
        "ha", "haa", "jeem", "kaaf", "khaa", "la", "laam", "meem", "nun", "ra",
        "saad", "seen", "sheen", "ta", "taa", "thaa", "thal", "toot", "waw",
        "ya", "yaa", "zay",
    )                                            # sorted; list index == class label

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
    select_metric: Literal["f1", "accuracy", "auc"] = "f1"  # checkpoint selection criterion
                                                # (NOT auc: a collapsed epoch can have high
                                                #  auc but ~random accuracy)
    num_workers: int = 4
    seed: int = 0
    device: str = "cuda"

    # ---- logging ----
    out_dir: str = "./runs"
    exp_name: str = "ppg_swint"

    def total_protos(self) -> int:
        return self.protos_per_class * self.num_classes
