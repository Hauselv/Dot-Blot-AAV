import numpy as np

from dotblot.preprocessing import detect_spot_polarity, invert_image, orient_image


def test_detect_spot_polarity_prefers_bright_spots():
    image = np.zeros((40, 40), dtype=np.float32)
    image[18:22, 18:22] = 10.0
    result = detect_spot_polarity(image)
    assert result["detected_polarity"] == "Spots hell"


def test_detect_spot_polarity_prefers_dark_spots():
    image = np.ones((40, 40), dtype=np.float32) * 10.0
    image[18:22, 18:22] = 0.0
    result = detect_spot_polarity(image)
    assert result["detected_polarity"] == "Spots dunkel"


def test_orient_image_inverts_for_dark_spots():
    image = np.ones((10, 10), dtype=np.float32)
    image[5, 5] = 0.0
    oriented, meta = orient_image(image, "Spots dunkel")
    assert meta["invert_applied"] is True
    assert oriented[5, 5] == oriented.max()
    assert np.allclose(oriented, invert_image(image))
