import numpy as np

from dotblot.detection import apply_crop_to_image, detect_spots, make_detection_image, sanitize_crop_box


def test_make_detection_image_reduces_smooth_background():
    yy, xx = np.mgrid[:80, :120]
    image = 10.0 + 0.08 * xx + 0.05 * yy
    image += 6.0 * np.exp(-(((xx - 55) ** 2 + (yy - 42) ** 2) / (2 * 3.0**2)))

    detection, background = make_detection_image(image.astype(np.float32), background_sigma=14.0)

    assert detection.shape == image.shape
    assert background.shape == image.shape
    assert detection[42, 55] > detection[10, 10]


def test_detect_spots_finds_small_spots_with_variable_spacing():
    image = np.zeros((120, 180), dtype=np.float32)
    yy, xx = np.mgrid[:120, :180]
    centers = [(28, 25), (63, 27), (109, 30), (35, 71), (86, 74), (145, 77)]
    amplitudes = [7.5, 6.5, 5.2, 7.0, 4.8, 6.2]
    sigmas = [2.2, 2.0, 1.8, 2.4, 1.7, 2.1]

    image += 4.0 + 0.03 * xx + 0.02 * yy
    for (x, y), amplitude, sigma in zip(centers, amplitudes, sigmas):
        image += amplitude * np.exp(-(((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2)))

    spots, detection_image, _ = detect_spots(
        oriented_image=image,
        background_sigma=16.0,
        min_sigma=1.0,
        max_sigma=5.0,
        num_sigma=10,
        threshold_rel=0.06,
        overlap=0.5,
        min_distance=8.0,
        adaptive_roi_enabled=True,
        adaptive_roi_threshold_rel=0.22,
        adaptive_roi_min_radius=2.5,
        adaptive_roi_max_radius=8.0,
    )

    assert len(spots) >= 5
    assert np.max(detection_image) > 0
    assert spots["roi_radius"].between(2.5, 8.0).all()

    detected_centers = spots.loc[:, ["center_x", "center_y"]].to_numpy(dtype=float)
    recovered = 0
    for x, y in centers:
        if np.min(np.hypot(detected_centers[:, 0] - x, detected_centers[:, 1] - y)) < 7.5:
            recovered += 1
    assert recovered >= 5


def test_crop_box_limits_detection_region():
    image = np.zeros((100, 150), dtype=np.float32)
    yy, xx = np.mgrid[:100, :150]
    image += 2.0
    image += 8.0 * np.exp(-(((xx - 25) ** 2 + (yy - 30) ** 2) / (2 * 2.5**2)))
    image += 9.0 * np.exp(-(((xx - 120) ** 2 + (yy - 70) ** 2) / (2 * 2.5**2)))

    crop_box = (0.0, 0.0, 70.0, 60.0)
    spots, _, _ = detect_spots(
        oriented_image=image,
        background_sigma=12.0,
        min_sigma=1.0,
        max_sigma=5.0,
        num_sigma=10,
        threshold_rel=0.08,
        overlap=0.5,
        min_distance=8.0,
        crop_box=crop_box,
        adaptive_roi_enabled=True,
        adaptive_roi_threshold_rel=0.22,
        adaptive_roi_min_radius=2.5,
        adaptive_roi_max_radius=8.0,
    )

    assert len(spots) >= 1
    assert (spots["center_x"] <= 70.0).all()


def test_apply_crop_to_image_zeros_outside_region():
    image = np.ones((20, 30), dtype=np.float32)
    cropped = apply_crop_to_image(image, (5.0, 4.0, 18.0, 12.0))
    sanitized = sanitize_crop_box((5.0, 4.0, 18.0, 12.0), image.shape)

    assert sanitized == (5, 4, 18, 12)
    assert float(cropped[:4, :].sum()) == 0.0
    assert float(cropped[:, :5].sum()) == 0.0
    assert float(cropped[4:12, 5:18].sum()) > 0.0
