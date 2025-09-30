# Asilo_1/fl/privacy/dp.py
import random

def clip_and_noise(delta_u: float, u_max: float, sigma: float = 0.0) -> float:
    du = max(0.0, min(u_max, delta_u))
    if sigma > 0:
        du += random.gauss(0.0, sigma)
    return max(0.0, du)
