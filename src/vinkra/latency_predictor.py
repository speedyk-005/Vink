import time
import warnings
from collections import deque

import numpy as np
from numpy.exceptions import RankWarning

# Minimum data points before outlier smoothing
_MIN_SMOOTH_SAMPLES = 2

# Minimum data points before curve fitting
_MIN_FIT_SAMPLES = 3


class LatencyPredictor:
    """A lean, structural predictor using only bounded Power Law fitting.

    Uses a Power Law model (y = a * x^b) to predict search latency based on
    the number of vectors in the index. Initial calibration measures raw
    BLAS performance, then online tuning refines parameters with actual
    runtime measurements.

    The model bounds exponents between 0.7 and 1.5 to keep predictions
    physically meaningful despite hardware jitter.
    """

    def __init__(self, dim: int = 128, window_size: int = 32):
        """Initialize latency predictor with Power Law model.

        Args:
            dim: Vector dimensionality for calibration search.
            window_size: Number of (n_vectors, latency) pairs to keep for online tuning.
        """
        self._dim = dim
        self.x_buffer = deque(maxlen=window_size)
        self.y_buffer = deque(maxlen=window_size)

        self._popt = np.array([1e-5, 1.0], dtype=np.float64)  # [a, b]
        self._calibrate_device()

    def _calibrate_device(self) -> None:
        """Calibrate the device by measuring raw BLAS performance."""
        # Scale by dim to keep work (vectors * dim) constant with empirical baseline
        # (128-dim, 20k-vecs)
        test_n = int((128 / self._dim) * 20000)

        vecs = np.random.randn(test_n, self._dim).astype(np.float32)
        q = np.random.randn(self._dim).astype(np.float32)

        self._calibration_search(vecs, q)  # Warm-up

        avg_ms = 0.0
        for _ in range(5):
            start = time.perf_counter()
            self._calibration_search(vecs, q)
            lat_ms = (time.perf_counter() - start) * 1000
            avg_ms = (avg_ms + lat_ms) / 2  # EMA-style blend

        avg_ms *= 0.9  # Account for Python overhead in actual usage.

        self._popt[0] = avg_ms / (test_n ** self._popt[1])

    def predict(self, n_vecs: int) -> float:
        """Predict latency for a given number of vectors in milliseconds."""
        return self._power_law(n_vecs, *self._popt)

    def tune(self, n_vecs: int, actual_lat: float) -> None:
        """Update model parameters with actual latency measurement.

        Args:
            n_vecs: Current number of vectors in the index.
            actual_lat: Actual measured latency in milliseconds.
        """
        # Guard against invalid values for logarithmic fitting
        self.x_buffer.append(max(n_vecs, 1))
        self.y_buffer.append(max(actual_lat, 1e-4))

        if len(self.x_buffer) < _MIN_FIT_SAMPLES:
            return

        x = np.asarray(self.x_buffer, dtype=np.float64)
        y = np.asarray(self.y_buffer, dtype=np.float64)

        try:
            # Linearize the power law:
            # log(y) = log(a) + b * log(x)
            log_x = np.log(x)
            log_y = np.log(y)

            # Least-squares fit in log-log space. All-identical n_vecs samples
            # produce an ill-conditioned design matrix; numpy's RankWarning is
            # expected and harmless since the fit is clipped below.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RankWarning)
                b, log_a = np.polyfit(log_x, log_y, deg=1)
            a = np.exp(log_a)

            self._popt = np.array(
                [
                    np.clip(a, 1e-10, 0.1),
                    np.clip(b, 0.7, 1.5),
                ],
                dtype=np.float64,
            )

        except (
            np.linalg.LinAlgError,
            FloatingPointError,
            ValueError,
        ):
            pass

    def _calibration_search(self, vectors: np.ndarray, query: np.ndarray) -> None:
        """Perform dummy search for timing calibration."""
        scores = vectors @ query
        _ = np.argpartition(scores, -10)[-10:]

    def _power_law(self, x: float, a: float, b: float) -> float:
        """Power Law function: y = a * x^b.

        Args:
            x: Input value (number of vectors).
            a: Scale coefficient.
            b: Exponent coefficient.

        Returns:
            Predicted latency value.
        """
        return a * np.power(x, b)


if __name__ == "__main__":  # pragma: no cover
    predictor = LatencyPredictor(dim=128)
    print(f"{'Step':<5} | {'N':<7} | {'Pred':<8} | {'Actual':<8} | {'Exp (b)':<5}")
    print("-" * 45)

    for i in range(25):
        n = (i + 1) * 10000
        v = np.random.randn(n, 128).astype(np.float32)
        q = np.random.randn(128).astype(np.float32)

        p = predictor.predict(n)
        t0 = time.perf_counter()
        predictor._calibration_search(v, q)
        act = (time.perf_counter() - t0) * 1000

        print(f"{n:<7} | {p:6.2f}ms | {act:6.2f}ms | {predictor._popt[1]:4.2f}")
        predictor.tune(n, act)

    print(f"\nFinal Predict for 200000 vecs: {predictor.predict(200000):.2f}ms")
