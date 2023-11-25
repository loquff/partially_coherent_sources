import numpy as np
from numpy.typing import ArrayLike


def random_phases(shape, seed=None) -> ArrayLike:
    """
    Generate an array of random phase values between 0 and 2π.

    Args:
        shape (tuple): The shape of the output array. 
        seed (int, optional): The seed for the random number generator. Defaults to None.

    Returns:
        (ArrayLike): An array of random phase values with the specified shape.
    """

    rng = np.random.RandomState(seed)
    phi = rng.uniform(0, 2 * np.pi, shape)
    return phi


def generate_masks(n_masks: int, weights: ArrayLike, fields: ArrayLike, method='phase_randomized', seed=None) -> ArrayLike:
    """
    Generate masks for the production of partially coherent light beams based on the given parameters.

    Args:
        n_masks (int): The number of masks to generate.
        weights (ArrayLike): The weights associated with each field.
        fields (ArrayLike): The fields used to generate the masks.
        method (str, optional): The method used to generate the masks. Defaults to 'phase_randomized'.
        seed (int, optional): The seed value for random number generation. Defaults to None.

    Returns:
        ArrayLike: The generated masks.

    Raises:
        AssertionError: If the method is not one of 'phase_randomized' or 'appearance_probability'.
        AssertionError: If the number of weights does not match the number of fields.
    """

    assert method in ['phase_randomized',
                      'appearance_probability'], 'Method must be one of "phase_randomized" or "appearance_probability"'
    assert weights.shape[0] == fields.shape[0], 'weights must have the same number of elements as fields'

    if method == 'phase_randomized':
        phases = random_phases((n_masks, fields.shape[0]), seed)
        kernel = np.reshape(fields, (fields.shape[0], -1))
        for n in range(fields.shape[0]):
            kernel[n] *= weights[n]
        masks = np.matmul(phases, kernel).reshape((n_masks, *fields.shape[1:]))

    elif method == 'appearance_probability':
        p = weights / np.sum(weights)
        samples = np.random.choice(fields.shape[0], size=n_masks, p=p)
        masks = np.stack([fields[n] for n in samples], axis=0)

    return masks
