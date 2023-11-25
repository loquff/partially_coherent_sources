import numpy as np
from partially_coherent_sources import *
import pytest


def test_random_phases_shape():
    shape = (5, 5)
    result = random_phases(shape)
    assert result.shape == shape


def test_random_phases_values_range():
    shape = (5, 5)
    result = random_phases(shape)
    assert (result >= 0).all() and (result <= 2 * np.pi).all()


def test_random_phases_seed():
    shape = (5, 5)
    seed = 1
    result1 = random_phases(shape, seed)
    result2 = random_phases(shape, seed)
    np.testing.assert_array_equal(result1, result2)


test_random_phases_shape()
test_random_phases_values_range()
test_random_phases_seed()


def test_generate_masks_phase_randomized():
    n_masks = 10
    weights = np.array([0.1, 0.2, 0.3, 0.4])
    fields = np.random.rand(4, 5, 5)
    seed = 1

    masks = generate_masks(n_masks, weights, fields,
                           method='phase_randomized', seed=seed)

    assert masks.shape == (n_masks, *fields.shape[1:])


def test_generate_masks_appearance_probability():
    n_masks = 10
    weights = np.array([0.1, 0.2, 0.3, 0.4])
    fields = np.random.rand(4, 5, 5)
    seed = 1

    masks = generate_masks(n_masks, weights, fields,
                           method='appearance_probability', seed=seed)

    assert masks.shape == (n_masks, *fields.shape[1:])


def test_generate_masks_invalid_method():
    n_masks = 10
    weights = np.array([0.1, 0.2, 0.3, 0.4])
    fields = np.random.rand(4, 5, 5)

    with pytest.raises(AssertionError):
        generate_masks(n_masks, weights, fields, method='invalid_method')


def test_generate_masks_mismatched_weights_fields():
    n_masks = 10
    weights = np.array([0.1, 0.2, 0.3])
    fields = np.random.rand(4, 5, 5)

    with pytest.raises(AssertionError):
        generate_masks(n_masks, weights, fields, method='phase_randomized')


test_generate_masks_phase_randomized()
test_generate_masks_appearance_probability()
test_generate_masks_invalid_method()
test_generate_masks_mismatched_weights_fields()


def test_docs_example():
    import numpy as np
    from partially_coherent_sources import generate_masks

    # Define the number of masks, weights, and fields
    n_masks = 10
    weights = np.array([0.1, 0.2, 0.3, 0.4])
    fields = np.random.rand(4, 5, 5)

    # Generate masks using the phase randomized method
    masks = generate_masks(n_masks, weights, fields, method='phase_randomized')

    # Generate masks using the appearance probability method
    masks = generate_masks(n_masks, weights, fields,
                           method='appearance_probability')


test_docs_example()
