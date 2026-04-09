import numpy as np

from dotblot.grid import auto_initialize_grid, generate_grid_centers, refine_grid_centers
from dotblot.types import GridConfig


def test_generate_grid_centers_without_rotation():
    config = GridConfig(
        rows=2,
        cols=2,
        anchor_x=10.0,
        anchor_y=20.0,
        pitch_x=30.0,
        pitch_y=40.0,
        rotation_deg=0.0,
        roi_radius=5.0,
    )
    centers = generate_grid_centers(config)
    expected = np.array([[10.0, 20.0], [40.0, 20.0], [10.0, 60.0], [40.0, 60.0]])
    assert np.allclose(centers, expected)


def test_auto_initialize_grid_finds_regular_peak_positions():
    image = np.zeros((120, 160), dtype=np.float32)
    xs = [30, 70, 110]
    ys = [25, 65]
    yy, xx = np.mgrid[:120, :160]
    for y in ys:
        for x in xs:
            image += 10.0 * np.exp(-(((xx - x) ** 2 + (yy - y) ** 2) / (2 * 4.0**2)))

    grid = auto_initialize_grid(image, rows=2, cols=3)
    assert abs(grid.anchor_x - 30) < 6
    assert abs(grid.anchor_y - 25) < 6
    assert abs(grid.pitch_x - 40) < 8
    assert abs(grid.pitch_y - 40) < 8


def test_refine_grid_centers_moves_center_towards_signal_peak():
    image = np.zeros((80, 80), dtype=np.float32)
    yy, xx = np.mgrid[:80, :80]
    image += 12.0 * np.exp(-(((xx - 42) ** 2 + (yy - 37) ** 2) / (2 * 3.5**2)))
    initial = np.array([[36.0, 32.0]], dtype=np.float32)

    refined = refine_grid_centers(image, initial, roi_radius=6.0, search_radius=10.0)
    initial_distance = np.hypot(initial[0, 0] - 42, initial[0, 1] - 37)
    refined_distance = np.hypot(refined[0, 0] - 42, refined[0, 1] - 37)

    assert refined_distance < initial_distance


def test_auto_initialize_grid_recovers_regular_spacing_with_weak_spots():
    image = np.zeros((140, 180), dtype=np.float32)
    yy, xx = np.mgrid[:140, :180]
    xs = [26, 62, 98, 134]
    ys = [24, 60, 96]
    amplitudes = np.array(
        [
            [11.0, 10.0, 2.2, 8.5],
            [10.5, 9.5, 1.3, 8.0],
            [10.0, 9.0, 0.9, 7.5],
        ],
        dtype=np.float32,
    )

    background = 1.5 + 0.015 * xx + 0.01 * yy
    image += background
    for row_index, y in enumerate(ys):
        for col_index, x in enumerate(xs):
            amplitude = amplitudes[row_index, col_index]
            image += amplitude * np.exp(-(((xx - x) ** 2 + (yy - y) ** 2) / (2 * 4.2**2)))

    grid = auto_initialize_grid(image, rows=3, cols=4)
    centers = generate_grid_centers(grid).reshape(3, 4, 2)

    assert abs(grid.anchor_x - xs[0]) < 7
    assert abs(grid.anchor_y - ys[0]) < 7
    assert abs(grid.pitch_x - 36) < 8
    assert abs(grid.pitch_y - 36) < 8
    assert abs(float(np.median(centers[:, 2, 0])) - xs[2]) < 8
    assert abs(float(np.median(centers[2, :, 1])) - ys[2]) < 8


def test_refine_grid_centers_handles_weak_spot_on_gradient_background():
    yy, xx = np.mgrid[:90, :90]
    image = 5.0 + 0.04 * xx + 0.03 * yy
    image += 4.0 * np.exp(-(((xx - 54) ** 2 + (yy - 39) ** 2) / (2 * 4.5**2)))
    initial = np.array([[47.0, 34.0]], dtype=np.float32)

    refined = refine_grid_centers(image.astype(np.float32), initial, roi_radius=7.0, search_radius=12.0)
    initial_distance = np.hypot(initial[0, 0] - 54, initial[0, 1] - 39)
    refined_distance = np.hypot(refined[0, 0] - 54, refined[0, 1] - 39)

    assert refined_distance < initial_distance
