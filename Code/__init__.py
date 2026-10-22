"""
MMA3001 Pork Rasher Packaging Inspection and Quality Assurance Package.
Student: Mikayla Holden
Student ID: 34990747
Unit: MMA3001 - Numerical Methods and Machine Learning
"""

__version__ = "1.0.0"
__author__ = "Mikayla Holden (34990747)"

from .validator import InputValidator, UnsupportedInputError
from .dataset import SyntheticDatasetGenerator, RoboflowDatasetLoader
from .baseline_model import BaselineRuleBasedDetector
from .ml_model import AdvancedMLDetector
from .numerical_integration import NumericalIntegrator2D
from .dynamic_ode import DynamicSystemSolver
from .profiler import ComputationalProfiler

__all__ = [
    "InputValidator",
    "UnsupportedInputError",
    "SyntheticDatasetGenerator",
    "RoboflowDatasetLoader",
    "BaselineRuleBasedDetector",
    "AdvancedMLDetector",
    "NumericalIntegrator2D",
    "DynamicSystemSolver",
    "ComputationalProfiler"
]
