"""Draw SVG forms using Fourier transformation."""
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate
import svgpathtools


def fdraw_coeffs(data: np.ndarray, coeffs_num: int = 101) -> None:
    """Estimate fourier coefficients from SVG."""
    if not isinstance(coeffs_num, int):
        coeffs_num = int(coeffs_num)

    t = np.linspace(0, 1, data.size)
    freqs = np.arange(coeffs_num) - (coeffs_num // 2)

    integrand = np.exp(-2j * np.pi * np.outer(freqs, t)) * data
    coeffs = scipy.integrate.trapz(integrand, dx=1 / data.size, axis=1)

    return coeffs.astype(np.complex64)


def fdraw_draw(coeffs: np.ndarray, num_points: int = 128) -> None:
    """Draw using fourier coefficients."""
    t = np.linspace(0, 1, num_points, dtype=np.complex64)
    freqs = np.arange(coeffs.size, dtype=np.complex64) - (coeffs.size // 2)

    imag = np.sum(np.exp(2j * np.pi * np.outer(t, freqs)) * coeffs, axis=1)

    _range_x = np.quantile(imag.real, (0, 1))
    _range_y = np.quantile(imag.imag, (0, 1))
    plt.xlim(_range_x + (_range_x[1] - _range_x[0]) * np.array([-0.1, 0.1]))
    plt.ylim(_range_y + (_range_y[1] - _range_y[0]) * np.array([-0.1, 0.1]))

    inds = np.round(np.linspace(0, imag.size, 200)).astype(int)

    _plot = True
    for start, end in zip(inds[:-1], inds[1:]):
        plt.scatter(imag.real[start:end],
                    imag.imag[start:end],
                    marker=".",
                    color="black",
                    s=0.5)
        try:
            plt.pause(1e-5)

        except KeyboardInterrupt:
            _plot = False
            break

    if _plot:
        plt.show()


def _test() -> None:
    def _load_data(img_id: int) -> np.ndarray:
        DATA_ID = {
            0: "./svgs/musical_note.svg",
            1: "./svgs/usp_logo.svg",
            2: "./svgs/coin.svg",
        }
    
        paths, attr = svgpathtools.svg2paths(DATA_ID[img_id])
    
        if img_id == 0:
            data = np.hstack([np.array(curve[:-1]) for curve in paths[0]])
    
        elif img_id >= 1:
            data = np.hstack([
                np.array(curve[:-1])
                for path in paths
                for curve in path
            ])
    
        if img_id == 2:
            data = data[::15]
    
        print("Data size:", data.size)
        data = data / np.max(np.abs(data))
    
        return data

    data = _load_data(2)
    coeffs = fdraw_coeffs(data, coeffs_num=data.size)
    fdraw_draw(coeffs, num_points=data.size * 32)


if __name__ == "__main__":
    _test()
