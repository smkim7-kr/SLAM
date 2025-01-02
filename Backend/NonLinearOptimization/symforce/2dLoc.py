# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

"""
Demonstrates solving a 2D localization problem with SymForce. The goal is for a robot
in a 2D plane to compute its trajectory given distance measurements from wheel odometry
and relative bearing angle measurements to known landmarks in the environment.
"""

# -----------------------------------------------------------------------------
# Set the default epsilon to a symbol
# -----------------------------------------------------------------------------
import symforce

# Set the default epsilon to a symbol to enable symbolic manipulation of numerical values.
# This is necessary for SymForce to use a nonzero epsilon to prevent singularities.
symforce.set_epsilon_to_symbol()

# -----------------------------------------------------------------------------
# Create initial Values
# -----------------------------------------------------------------------------
import numpy as np

from symforce import typing as T
from symforce.values import Values


def build_initial_values() -> T.Tuple[Values, int, int]:
    """
    Creates a Values with numerical values for the constants in the problem, and initial guesses
    for the optimized variables
    """
    num_poses = 5
    num_landmarks = 6

    initial_values = Values(
        # List of initial pose values, each set to identity
        poses=[sf.Pose2.identity()] * num_poses,
        # List of initial landmark positions
        landmarks=[sf.V2(-2, 2), sf.V2(1, -3), sf.V2(5, 2), sf.V2(4, 7), sf.V2(0, 1), sf.V2(4, 4)],
        # List of distance measurements (pose t to pose t+1)
        distances=[1.7, 1.4, 2.0, 4.4],
        # List of angle measurements in radians (angles[pose t][landmark l])
        angles=np.deg2rad([[55, 245, -35, 20, -11, 45], [95, 220, -20, 20, -11, 45], [125, 220, -20, 20, -11, 45], \
            [55, 245, -35, 20, -11, 45], [95, 220, -20, 20, -11, 45], [125, 220, -20, 20, -11, 45]]).tolist(),
        # Symbolic value for numeric epsilon
        epsilon=sf.numeric_epsilon,
    )

    return initial_values, num_poses, num_landmarks


# -----------------------------------------------------------------------------
# Define residual functions
# -----------------------------------------------------------------------------
import symforce.symbolic as sf


def bearing_residual(
    pose: sf.Pose2, landmark: sf.V2, angle: sf.Scalar, epsilon: sf.Scalar
) -> sf.V1:
    """
    Residual from a relative bearing measurement of a 2D pose to a landmark.
    """
    # Calculate the relative position of the landmark in the body frame
    # landmark is originally at world frame
    # t_body is expressed in robot's body frame
    t_body = pose.inverse() * landmark
    # Predict the angle to the landmark
    predicted_angle = sf.atan2(t_body[1], t_body[0], epsilon=epsilon)
    # Calculate the residual by wrapping the predicted angle and subtracting the actual angle
    # sf.wrap_angle ensures the angle is at specific range (prevent discontinutieis of sinosidul funcation)
    return sf.V1(sf.wrap_angle(predicted_angle - angle))


def odometry_residual(
    pose_a: sf.Pose2, pose_b: sf.Pose2, dist: sf.Scalar, epsilon: sf.Scalar
) -> sf.V1:
    """
    Residual from the scalar distance between two poses.
    """ 
    # sf.Pose2.t is the position x,y
    # epsilon used for numerical stabbility (prevent singularities)
    # when pose_a and pose_b is very small, can cause singularites
    return sf.V1((pose_b.t - pose_a.t).norm(epsilon=epsilon) - dist)


# -----------------------------------------------------------------------------
# Create a set of factors to represent the full problem
# -----------------------------------------------------------------------------
from symforce.opt.factor import Factor


def build_factors(num_poses: int, num_landmarks: int) -> T.Iterator[Factor]:
    """
    Build factors for a problem of the given dimensionality.
    """
    for i in range(num_poses - 1):
        yield Factor(
            residual=odometry_residual,
            keys=[f"poses[{i}]", f"poses[{i + 1}]", f"distances[{i}]", "epsilon"],
        )

    for i in range(num_poses):
        for j in range(num_landmarks):
            yield Factor(
                residual=bearing_residual,
                keys=[f"poses[{i}]", f"landmarks[{j}]", f"angles[{i}][{j}]", "epsilon"],
            )


# -----------------------------------------------------------------------------
# Instantiate, optimize, and visualize
# -----------------------------------------------------------------------------
from symforce.opt.optimizer import Optimizer


def main() -> None:
    # Create a problem setup and initial guess
    initial_values, num_poses, num_landmarks = build_initial_values()

    # Create factors
    factors = build_factors(num_poses=num_poses, num_landmarks=num_landmarks)

    # Select the keys to optimize - the rest will be held constant
    optimized_keys = [f"poses[{i}]" for i in range(num_poses)]

    # Create the optimizer
    optimizer = Optimizer(
        factors=factors,
        optimized_keys=optimized_keys,
        debug_stats=True,  # Return problem stats for every iteration
        params=Optimizer.Params(verbose=True),  # Customize optimizer behavior
    )

    # Solve and return the result
    result = optimizer.optimize(initial_values)

    # Print some values
    print(f"Num iterations: {len(result.iterations) - 1}")
    print(f"Final error: {result.error():.6f}")
    print(f"Status: {result.status}")

    for i, pose in enumerate(result.optimized_values["poses"]):
        print(f"Pose {i}: t = {pose.position()}, heading = {pose.rotation().to_tangent()[0]}")

    # Plot the result
    # TODO(hayk): mypy gives the below error, but a relative import also doesn't work.
    # Skipping analyzing "symforce.examples.robot_2d_localization.plotting":
    #     found module but no type hints or library stubs
    from symforce.examples.robot_2d_localization.plotting import plot_solution

    plot_solution(optimizer, result)


if __name__ == "__main__":
    main()