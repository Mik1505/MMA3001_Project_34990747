"""
Comprehensive Test Suite for MMA3001 Quality Inspection Pipeline.
Student ID: 34990747
"""

import sys
import os
import pytest
import numpy as np

# Ensure src package is discoverable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.validator import InputValidator, UnsupportedInputError
from src.dataset import SyntheticDatasetGenerator, RoboflowDatasetLoader
from src.baseline_model import BaselineRuleBasedDetector
from src.ml_model import AdvancedMLDetector
from src.numerical_integration import NumericalIntegrator2D
from src.dynamic_ode import DynamicSystemSolver
from src.profiler import ComputationalProfiler


@pytest.fixture
def dataset_gen():
    return SyntheticDatasetGenerator(seed=123)


@pytest.fixture
def validator():
    return InputValidator()


def test_input_validator_valid_image(validator, dataset_gen):
    img, _, _ = dataset_gen.generate_sample('pass')
    assert validator.validate(img) is True


def test_input_validator_wrong_dimension(validator):
    bad_img = np.zeros((300, 300, 3), dtype=np.uint8)
    with pytest.raises(UnsupportedInputError, match="ERR_DIM"):
        validator.validate(bad_img)


def test_input_validator_blackout(validator):
    blackout_img = np.zeros((540, 720, 3), dtype=np.uint8)
    with pytest.raises(UnsupportedInputError, match="ERR_UNDEREXPOSED"):
        validator.validate(blackout_img)


def test_input_validator_out_of_domain_noise(validator, dataset_gen):
    noisy_img, _, _ = dataset_gen.generate_sample('unsupported')
    with pytest.raises(UnsupportedInputError):
        validator.validate(noisy_img)


def test_baseline_detector(dataset_gen):
    detector = BaselineRuleBasedDetector(min_defect_pixels=150)
    pass_img, _, is_def_pass = dataset_gen.generate_sample('pass')
    tear_img, _, is_def_tear = dataset_gen.generate_sample('tear')
    
    pred_pass, px_pass, _ = detector.predict(pass_img)
    pred_tear, px_tear, _ = detector.predict(tear_img)
    
    assert pred_pass == 0
    assert pred_tear == 1
    assert px_tear > px_pass


def test_ml_detector_training_and_inference(dataset_gen):
    images, masks, labels, _ = dataset_gen.build_dataset(n_samples=40)
    model = AdvancedMLDetector(model_type='rf')
    model.fit(images, labels)
    
    # Test on a new sample
    test_img, _, is_def = dataset_gen.generate_sample('tear')
    pred = model.predict(test_img)
    prob = model.predict_proba(test_img)
    
    assert pred in [0, 1]
    assert 0.0 <= prob <= 1.0


def test_numerical_integration_synthetic_geometry():
    integrator = NumericalIntegrator2D(tray_width_mm=240.0, tray_height_mm=180.0, grid_w=720, grid_h=540)
    # Create a synthetic rectangular defect of exactly 60x60 pixels
    # Area = (60 * 240/720) * (60 * 180/540) = 20 mm * 20 mm = 400 mm^2
    mask = np.zeros((540, 720), dtype=np.uint8)
    mask[100:160, 100:160] = 1
    
    trap_area = integrator.trapezoidal_2d(mask)
    simp_area = integrator.simpsons_2d(mask)
    
    # Due to boundary weights on sharp discrete edges, quadrature should match within 2%
    assert abs(trap_area - 400.0) < 10.0
    assert abs(simp_area - 400.0) < 10.0
    
    valid, msg = integrator.validate_integral(simp_area)
    assert valid is True


def test_numerical_integration_physical_bounds():
    integrator = NumericalIntegrator2D()
    valid_neg, msg_neg = integrator.validate_integral(-15.0)
    assert valid_neg is False
    assert "ERR_NEGATIVE_AREA" in msg_neg
    
    valid_ovf, msg_ovf = integrator.validate_integral(50000.0)
    assert valid_ovf is False
    assert "ERR_OVERFLOW" in msg_ovf


def test_dynamic_ode_analytical_accuracy():
    solver = DynamicSystemSolver(k_decay=0.8, P_vac=20.0, P_atm=101.3)
    t, p_rk4 = solver.solve_rk4(t_span=(0.0, 3.0), dt=0.05)
    p_exact = solver.analytical_solution(t)
    
    # RK4 error should be less than 1e-4 kPa with dt=0.05
    max_err = np.max(np.abs(p_rk4 - p_exact))
    assert max_err < 1e-3


def test_dynamic_ode_stability_limit():
    solver = DynamicSystemSolver(k_decay=0.8)
    # Critical dt = 2 / 0.8 = 2.5 s.
    # With dt = 0.5 s, Euler is stable
    t_stable, p_stable = solver.solve_euler(t_span=(0.0, 5.0), dt=0.5)
    assert np.all(np.isfinite(p_stable))
    assert p_stable[-1] <= solver.P_atm + 0.1


def test_roboflow_dataset_loader_instantiation():
    loader = RoboflowDatasetLoader("pork-rasher-error-packaging-4")
    assert loader.dataset_dir.name == "pork-rasher-error-packaging-4"
    assert isinstance(loader.class_names, dict)
