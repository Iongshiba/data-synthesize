from __future__ import annotations

import math
import numpy as np
import sympy as sp
from typing import Callable, Iterable, Any
from utils import *
from rendering.world import Rotate, Scale, Translate, Composite
from shape import Equation

AnimationFn = Callable[[object, float], None]


def gradient_descent(
    equation: Equation,
    start_pos: Iterable[float, float, float],
    ball_radius: float,
    optimizer: str = "SGD",
    learning_rate: float = 5.0,
    momentum: float = 1.0,
    epsilon: float = 1e-4,
    decay_rate: float = 0.99,
    beta1: float = 0.99,  # first moment
    beta2: float = 0.999,  # second moment
    weight_decay: float = 0.01,  # L2 penalty for AdamW
    min_gradient: float = 0.001,
    max_gradient: float = 0.03,
):
    dx, dy = make_numpy_deri(equation.expression)
    func = make_numpy_func(equation.expression)
    x, y, z = start_pos
    velocity = 0
    accumulated_grad = 0
    time_step = 0

    def update(transform: Composite, dt: float) -> None:
        nonlocal x, y, z, velocity, accumulated_grad, time_step

        if isinstance(transform[0], Translate):
            translate = transform[0]
            rotate = transform[1]
        else:
            translate = transform[1]
            rotate = transform[0]

        x_grad = dx(x, y)
        y_grad = dy(x, y)
        xy_grad = np.array([x_grad, y_grad], dtype=np.float32)
        xy_grad_norm = np.linalg.norm(xy_grad)

        # gradient clipping
        if xy_grad_norm > max_gradient:
            xy_grad = xy_grad * (max_gradient / xy_grad_norm)
            xy_grad_norm = max_gradient

        # fmt: off
        if optimizer == "momentum":
            velocity = momentum * velocity + learning_rate * xy_grad * dt
            displacement = -velocity

        elif optimizer == "adagrad":
            accumulated_grad += xy_grad**2
            adapted_learning_rate = learning_rate / (np.sqrt(accumulated_grad) + epsilon)
            displacement = -adapted_learning_rate * xy_grad * dt

        elif optimizer == "rmsdrop":
            accumulated_grad = decay_rate * accumulated_grad + (1 - decay_rate) * xy_grad**2
            adapted_learning_rate = learning_rate / (np.sqrt(accumulated_grad) + epsilon)
            displacement = -adapted_learning_rate * xy_grad * dt * 0.1  # Scale down displacement
        elif optimizer == "adam":
            time_step += 1
            velocity = beta1 * velocity + (1 - beta1) * xy_grad
            accumulated_grad = beta2 * accumulated_grad + (1 - beta2) * xy_grad**2
            
            # bias correction
            velocity_corrected = velocity / (1 - beta1**time_step)
            accumulated_grad_corrected = accumulated_grad / (1 - beta2**time_step)
            
            displacement = -learning_rate * velocity_corrected / (np.sqrt(accumulated_grad_corrected) + epsilon) * dt

        elif optimizer == "adamw":
            time_step += 1
            # AdamW: Adam with decoupled weight decay
            velocity = beta1 * velocity + (1 - beta1) * xy_grad
            accumulated_grad = beta2 * accumulated_grad + (1 - beta2) * xy_grad**2
            
            # bias correction
            velocity_corrected = velocity / (1 - beta1**time_step)
            accumulated_grad_corrected = accumulated_grad / (1 - beta2**time_step)
            
            # AdamW: weight decay applied directly to parameters, not gradient
            displacement = -learning_rate * velocity_corrected / (np.sqrt(accumulated_grad_corrected) + epsilon) * dt
            # Apply weight decay directly to position (decoupled from gradient)
            position_vector = np.array([x, y], dtype=np.float32)
            displacement = displacement - learning_rate * weight_decay * position_vector * dt

        elif optimizer == "adarpop":
            time_step += 1
            # ADArpop: Adaptive learning rate with momentum
            velocity = beta1 * velocity + (1 - beta1) * xy_grad
            accumulated_grad = beta2 * accumulated_grad + (1 - beta2) * xy_grad**2
            
            # bias correction
            velocity_corrected = velocity / (1 - beta1**time_step)
            accumulated_grad_corrected = accumulated_grad / (1 - beta2**time_step)
            
            # ADArpop: adaptive clipping based on accumulated gradient
            adaptive_clip = np.sqrt(accumulated_grad_corrected) * max_gradient
            clipped_velocity = np.clip(velocity_corrected, -adaptive_clip, adaptive_clip)
            
            displacement = -learning_rate * clipped_velocity / (np.sqrt(accumulated_grad_corrected) + epsilon) * dt

        else:
            # SGD: Simple gradient descent
            displacement = -learning_rate * xy_grad * dt
        # fmt: on

        # Calculate new position
        new_x = x + displacement[0]
        new_y = y + displacement[1]
        new_z = (
            (func(new_x, new_y) - equation.Z_min)
            / (equation.Z_max - equation.Z_min)
            * 10
        )

        # Calculate movement vector
        move_direction = np.array([new_x - x, new_y - y, new_z - z])
        move_distance = np.linalg.norm(move_direction)

        # Roll (only when gradient is significant and ball is moving)
        if xy_grad_norm > min_gradient * 0.01 and move_distance > 1e-6:
            # Update position
            z_scale = 10.0 / (equation.Z_max - equation.Z_min)
            grad = np.array(
                [-x_grad * z_scale, -y_grad * z_scale, 1.0], dtype=np.float32
            )
            grad = grad / np.linalg.norm(grad)
            translate.x = new_x + ball_radius * grad[0]
            translate.y = new_y + ball_radius * grad[1]
            translate.z = new_z + ball_radius * grad[2]

            # For rolling motion, rotation axis is perpendicular to movement in XY plane
            move_direction_2d = np.array([move_direction[0], move_direction[1]])
            move_dist_2d = np.linalg.norm(move_direction_2d)

            if move_dist_2d > 1e-6:
                # Rotation axis: perpendicular to 2D movement (cross with Z-axis)
                rotation_axis = np.cross(
                    move_direction_2d / move_dist_2d, np.array([0.0, 0.0, 1.0])
                )
                axis_norm = np.linalg.norm(rotation_axis)

                if axis_norm > 1e-6:
                    rotation_axis = rotation_axis / axis_norm

                    # Calculate rotation angle: arc length / radius
                    rotation_angle_radians = move_distance / ball_radius
                    rotation_angle_degrees = np.degrees(rotation_angle_radians)

                    # Accumulate rotation
                    rotate.axis = tuple(rotation_axis)
                    rotate.angle -= rotation_angle_degrees

            # Update gradient
            x = new_x
            y = new_y
            z = new_z

    return update


def infinite_spin(speed: float = 1.0) -> AnimationFn:
    def update(transform: Rotate, dt: float) -> None:
        transform.angle = float((transform.angle + dt * speed) % 360.0)

    return update


def circular_orbit(
    phase: float = 0.0, speed: float = 1.0, radius: float = 1.0, axis: str = "xy"
) -> AnimationFn:
    theta = phase
    axis = axis.lower()
    axes = {
        "xy": (0, 1),
        "xz": (0, 2),
        "yz": (1, 2),
    }

    def update(transform: Translate, dt: float) -> None:
        nonlocal theta
        theta = (theta + dt * speed) % (2 * math.pi)
        coords = [transform.x, transform.y, transform.z]
        first, second = axes[axis]
        coords[first] = math.cos(theta) * radius
        coords[second] = math.sin(theta) * radius
        transform.x, transform.y, transform.z = coords

    return update


def ping_pong_translation(
    axis: str = "y",
    amplitude: float = 1.0,
    speed: float = 1.0,
    center: float = 0.0,
) -> AnimationFn:
    axis = axis.lower()

    offset = 0.0

    def update(transform: Translate, dt: float) -> None:
        nonlocal offset
        offset = (offset + dt * speed) % (2 * math.pi)
        value = center + math.sin(offset) * amplitude
        setattr(transform, axis, value)

    return update


def pulse_scale(
    minimum: float = 0.5,
    maximum: float = 1.5,
    speed: float = 1.0,
) -> AnimationFn:
    if minimum > maximum:
        minimum, maximum = maximum, minimum

    phase = 0.0
    span = maximum - minimum

    def update(transform: Scale, dt: float) -> None:
        nonlocal phase
        phase = (phase + dt * speed) % (2 * math.pi)
        factor = minimum + (math.sin(phase) * 0.5 + 0.5) * span
        transform.x = transform.y = transform.z = factor

    return update


__all__ = [
    "AnimationFn",
    "gradient_descent",
    "infinite_spin",
    "circular_orbit",
    "ping_pong_translation",
    "pulse_scale",
]
