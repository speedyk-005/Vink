import time
from collections import deque

import numpy as np
from scipy.optimize import curve_fit


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
            dim (int): Vector dimensionality for calibration search.
            window_size (int): Number of (n_vectors, latency) pairs to keep for online tuning.
        """
        self._dim = dim
        self.x_buffer = deque(maxlen=window_size)
        self.y_buffer = deque(maxlen=window_size)

        self._popt = [1e-5, 1.0]  # [a, b] -> y = a * x^b
        self._calibrate_device()

    def _calibrate_device(self) -> None:
        """Calibrate the device by measuring raw BLAS performance."""
        # Scale by dim to keep work (vectors * dim) constant with empirical baseline (128-dim, 20k-vecs)
        test_n = int((128 / self._dim) * 20000)

        vecs = np.random.randn(test_n, self._dim).astype(np.float32)
        q = np.random.randn(self._dim).astype(np.float32)

        self._calibration_search(vecs, q)  # Warm-up

        avg_ms = 0
        for _ in range(5):
            start = time.perf_counter()
            self._calibration_search(vecs, q)
            lat_ms = (time.perf_counter() - start) * 1000
            avg_ms = (avg_ms + lat_ms) / 2  # EMA-style blend

        avg_ms = avg_ms * 0.9  # Account for Python overhead in actual usage.

        self._popt[0] = avg_ms / (test_n ** self._popt[1])

    def predict(self, n_vecs: int) -> float:
        """Predict latency for a given number of vectors in milliseconds."""
        return self._power_law(n_vecs, *self._popt)

    def tune(self, n_vecs: int, actual_lat: float) -> None:
        """Update model parameters with actual latency measurement.

        Args:
            n_vecs (int): Current number of vectors in the index.
            actual_lat (float): Actual measured latency in milliseconds.
        """
        # Outlier smoothing: blend with prediction to avoid over-reaction to spikes
        if len(self.x_buffer) >= 2:
            pred = self.predict(n_vecs)
            if actual_lat > pred * 2:
                # Blend: 70% predicted, 30% actual - reduces spike impact
                actual_lat = pred * 0.7 + actual_lat * 0.3

        self.x_buffer.append(n_vecs)
        self.y_buffer.append(max(actual_lat, 1e-4))

        if len(self.x_buffer) >= 3:
            try:
                # Bounds keep the 'Physics' sane despite hardware jitter
                lower_bounds = [1e-10, 0.7]
                upper_bounds = [0.1, 1.5]

                new_popt, _ = curve_fit(
                    self._power_law,
                    list(self.x_buffer),
                    list(self.y_buffer),
                    p0=self._popt,
                    bounds=(lower_bounds, upper_bounds),
                    method='trf',
                    maxfev=50
                )
                self._popt = new_popt
            except Exception:
                pass

    def _calibration_search(self, vectors: np.ndarray, query: np.ndarray) -> None:
        """Perform dummy search for timing calibration."""
        scores = vectors @ query
        _ = np.argpartition(scores, -10)[-10:]

    def _power_law(self, x: float, a: float, b: float) -> float:
        """Power Law function: y = a * x^b.

        Args:
            x (float): Input value (number of vectors).
            a (float): Scale coefficient.
            b (float): Exponent coefficient.

        Returns:
            float: Predicted latency value.
        """
        return a * np.power(x, b)


if __name__ == "__main__":  # pragma: no cover
    predictor = LatencyPredictor(dim=128)
    print(f"{'Step':<5} | {'N':<7} | {'Pred':<8} | {'Actual':<8} | {'Exp (b)':<5}")
    print("-" * 45)

    for i in range(15):
        n = (i + 1) * 10000
        v = np.random.randn(n, 128).astype(np.float32)
        q = np.random.randn(128).astype(np.float32)

        p = predictor.predict(n)
        t0 = time.perf_counter()
        predictor._calibration_search(v, q)
        act = (time.perf_counter() - t0) * 1000

        print(f"{i+1:<5} | {n:<7} | {p:6.2f}ms | {act:6.2f}ms | {predictor._popt[1]:4.2f}")
        predictor.tune(n, act)

    print(f"\nFinal Predict for 200000 vecs: {predictor.predict(200000):.2f}ms")
