import math
import numpy as np
import matplotlib.pyplot as plt

# https://github.com/igorbasko01/mediapipe-jitter

class LowPassFilter:
    def __init__(self, alpha):
        self.alpha = alpha
        self.last_raw_value = None

    def apply_with_alpha(self, value, alpha):
        if self.last_raw_value is None:
            self.last_raw_value = value
        else:
            self.last_raw_value = alpha * value + \
                (1 - alpha) * self.last_raw_value
        return self.last_raw_value

    def apply(self, value):
        return self.apply_with_alpha(value, self.alpha)

class OneEuroFilter:
    def __init__(self, frequency, min_cutoff=1.0, beta=0.0, derivate_cutoff=1.0, to_print=False):
        if frequency <= 0:
            raise ValueError("Frequency should be > 0")
        if min_cutoff <= 0:
            raise ValueError("Min cutoff should be > 0")
        if derivate_cutoff <= 0:
            raise ValueError("Derivate cutoff should be > 0")

        self.frequency = frequency
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.derivate_cutoff = derivate_cutoff

        self.x = LowPassFilter(self.alpha(self.min_cutoff))
        self.dx = LowPassFilter(self.alpha(self.derivate_cutoff))

        self.last_time = np.iinfo(np.int64).min

        self.to_print = to_print

    def alpha(self, cutoff):
        te = 1.0 / self.frequency
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def apply(self, value, timestamp, value_scale=1.0):
        new_timestamp = timestamp

        if self.last_time >= new_timestamp:
            print("New timestamp is equal or less than the last one.")
            return value

        if self.last_time != 0 and new_timestamp != 0:
            self.frequency = 1.0 / (new_timestamp - self.last_time)
        self.last_time = new_timestamp

        dvalue = self.x.last_raw_value if self.x.last_raw_value is not None else 0.0
        dvalue = (value - dvalue) * value_scale * self.frequency

        edvalue = self.dx.apply_with_alpha(
            dvalue, self.alpha(self.derivate_cutoff))

        cutoff = self.min_cutoff + self.beta * abs(edvalue)

        result = self.x.apply_with_alpha(value, self.alpha(cutoff))

        # if self.to_print:
        #     print(f"original: {value}, new: {result}")

        return result