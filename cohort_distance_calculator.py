import numpy as np
import math
from scipy.optimize import fsolve
from cohort_utils import distance_to_length, length_to_distance


# Define the function to solve for d
def intersection_equation(d, r, R, A_part):
    term1 = r**2 * np.arccos((d**2 + r**2 - R**2) / (2 * d * r))
    term2 = R**2 * np.arccos((d**2 + R**2 - r**2) / (2 * d * R))
    term3 = 0.5 * np.sqrt((-d + r + R) * (d + r - R) * (d - r + R) * (d + r + R))

    return term1 + term2 - term3 - A_part


def solve_for_d_by_coverage(cohort_radius, query_radius, coverage):
    r = distance_to_length(cohort_radius)
    R = distance_to_length(query_radius)
    proportion = coverage
    A_r = np.pi * r**2  # Area of smaller circle
    A_part = A_r * proportion  # The required intersection area

    # Solve for d
    d_initial_guess = (r + R) / 2  # A reasonable starting point
    d_solution = fsolve(intersection_equation, d_initial_guess, args=(r, R, A_part))

    return length_to_distance(d_solution[0])  # Convert back to distance metric