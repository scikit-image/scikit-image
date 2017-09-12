import pytest

from ..generate_shapes import generate_shapes


def test_generates_color_images_with_correct_shape():
    image, _ = generate_shapes((128, 128), max_shapes=10)
    assert image.shape == (128, 128, 3)


def test_generates_gray_images_with_correct_shape():
    image, _ = generate_shapes(
        (4567, 123), min_shapes=3, max_shapes=20, gray=True)
    assert image.shape == (4567, 123, 1)


def test_generates_correct_bounding_boxes_for_rectangles():
    image, labels = generate_shapes(
        (128, 128),
        max_shapes=1,
        shape='rectangle',
        min_pixel_intensity=1,
        random_seed=42)
    assert len(labels) == 1
    label = labels[0]
    crop = image[label.y1:label.y2, label.x1:label.x2]

    # The crop is filled.
    assert (crop < 255).all()

    # The crop is complete.
    image[label.y1:label.y2, label.x1:label.x2] = 255
    assert (image == 255).all()


def test_generates_correct_bounding_boxes_for_triangles():
    image, labels = generate_shapes(
        (128, 128),
        max_shapes=1,
        shape='triangle',
        min_pixel_intensity=1,
        random_seed=42)
    # assert len(labels) == 1
    label = labels[0]
    crop = image[label.y1:label.y2, label.x1:label.x2]

    # The crop is filled.
    assert (crop < 255).any()

    # The crop is complete.
    image[label.y1:label.y2, label.x1:label.x2] = 255
    assert (image == 255).all()


def test_generates_correct_bounding_boxes_for_circles():
    image, labels = generate_shapes(
        (43, 44),
        max_shapes=1,
        min_size=20,
        max_size=20,
        shape='circle',
        min_pixel_intensity=1,
        random_seed=42)
    assert len(labels) == 1
    label = labels[0]
    crop = image[label.y1:label.y2, label.x1:label.x2]

    # The crop is filled.
    assert (crop < 255).any()

    # The crop is complete.
    image[label.y1:label.y2, label.x1:label.x2] = 255
    assert (image == 255).all()


def test_generate_circle_throws_when_size_too_small():
    with pytest.raises(ValueError):
        generate_shapes(
            (64, 128), max_shapes=1, min_size=1, max_size=1, shape='circle')


def test_generate_triangle_throws_when_size_too_small():
    with pytest.raises(ValueError):
        generate_shapes(
            (128, 64), max_shapes=1, min_size=1, max_size=1, shape='triangle')


def test_can_generate_one_by_one_rectangle():
    image, labels = generate_shapes(
        (50, 128),
        max_shapes=1,
        min_size=1,
        max_size=1,
        shape='rectangle',
        min_pixel_intensity=1)
    assert len(labels) == 1
    label = labels[0]
    crop = image[label.y1:label.y2, label.x1:label.x2]
    assert (crop < 255).sum() == 3  # rgb


def test_throws_when_min_pixel_intensity_out_of_range():
    with pytest.raises(ValueError):
        generate_shapes((1000, 1234), max_shapes=1, min_pixel_intensity=256)
    with pytest.raises(ValueError):
        generate_shapes((2, 2), max_shapes=1, min_pixel_intensity=-1)


def test_returns_empty_labels_and_white_image_when_cannot_fit_shape():
    # The circle will never fit this.
    image, labels = generate_shapes(
        (10000, 10000), max_shapes=1, min_size=10000, shape='circle')
    assert len(labels) == 0
    assert (image == 255).all()


def test_generate_shapes_is_reproducible_with_seed():
    random_seed = 42
    labels = []
    for _ in range(5):
        _, l = generate_shapes(
            (128, 128), max_shapes=5, random_seed=random_seed)
        labels.append(l)
    assert all(other == labels[0] for other in labels[1:])


def test_generates_white_image_when_min_pixel_intensity_255():
    image, labels = generate_shapes(
        (128, 128), max_shapes=3, min_pixel_intensity=255, random_seed=42)
    assert len(labels) > 0
    assert (image == 255).all()
