from enum import Enum

class PerformanceMetric(Enum):
    PRECISION = "precision"
    RECALL = "recall"
    F1 = "f1"

class ValidationMetric(Enum):
    B3 = "b3"
    PAIRWISE = "pairwise"