# setup = '''
import numpy as np
from numba import njit, jit, prange


def norm_func(min_val: int, max_val: int):
    def wrapper(val: int):
        return (val - min_val) / (max_val - min_val)

    return wrapper


def range_mesh(x_values: np.ndarray, y_values: np.ndarray, range_vals: tuple[int, int]) -> (np.ndarray, np.ndarray):
    aspect_ratio: float = len(x_values) / len(y_values)

    # prepare normalization
    max_val: int = max(y_values)
    min_val: int = min(y_values)
    y_normfunc = np.vectorize(norm_func(min_val, max_val))
    max_val, min_val = max(x_values), min(x_values)
    x_normfunc = np.vectorize(norm_func(min_val, max_val))

    # normalize values
    y_normalized: np.ndarray = y_normfunc(y_values)
    x_normalized: np.ndarray = x_normfunc(x_values)
    x_normalized = x_normalized * aspect_ratio

    # scale the normalized values
    scale: int = max(range_vals) - min(range_vals)
    y_scaled: np.ndarray = np.vectorize(lambda x: x * scale + min(range_vals))(y_normalized)
    x_scaled: np.ndarray = np.vectorize(lambda x: x * scale + min(range_vals) * aspect_ratio)(x_normalized)

    x_mesh, y_mesh = np.meshgrid(x_scaled, y_scaled)
    return x_mesh, y_mesh


# @njit(fastmath=True)
def mandelbrott(c: np.ndarray[complex], max_iter: int = 100) -> np.ndarray:
    # z array for iteration
    z: np.ndarray = np.zeros(c.shape, dtype=np.complex128)
    # array to count number of iterations before breaking
    # assume all to have made it to max iters at first
    niters: np.ndarray = np.full(c.shape, max_iter, dtype=int)
    for ix in range(max_iter):
        # select all values that have not been changed yet
        mask = niters == max_iter
        # calculate new z
        z[mask] = z[mask] ** 2 + c[mask]
        # if abs(z) value is > 2 set the iter_val to current iteration
        niters[mask & ((z.real ** 2 + z.imag ** 2) > 4.0)] = ix + 1

    # return the iter values normalized
    return niters / max_iter


# @njit(fastmath=True)
def Return_Mandelbrott(width: int, height: int, offsetval: float = 1.3, zoomval: float = 2.2) -> np.ndarray:
    width: np.ndarray = np.arange(width)
    height: np.ndarray = np.arange(height)

    # calc offset and zoom
    offset = np.array([offsetval * len(width), len(height)]) // 2
    zoom = zoomval / len(height)
    x = (width - offset[0]) * zoom
    y = (height - offset[1]) * zoom

    # make complex matrix
    complexgrid = x + 1j * y[:, None]
    # mandelgrid = np.zeros(complexgrid.shape, dtype=float)
    #
    # i, j = complexgrid.shape
    # for ix in range(i):
    #     for jx in range(j):
    #         res = mandelbrott(complexgrid[ix, jx])
    #         mandelgrid[ix, jx] = res
    return mandelbrott(complexgrid)
    # print(mandelgrid)
    # print(np.vectorize(mandelbrott)(complexgrid))


if __name__ == '__main__':
    import timeit
    import cProfile

    Return_Mandelbrott(200, 100)
    # print(min(timeit.repeat(setup=setup, stmt='test()', number=2, repeat=2))/2)
    # cProfile.run(setup+'test()', sort='tottime')
