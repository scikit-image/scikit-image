import numpy as np
import pytest

from ..generate_shapes import generate_shapes


def test_generates_color_images_with_correct_shape():
    image, _ = generate_shapes(width=128, height=128, max_shapes=10)
    assert image.shape == (128, 128, 3)


def test_generates_gray_images_with_correct_shape():
    image, _ = generate_shapes(width=123,
                               height=4567,
                               max_shapes=200,
                               gray=True)
    assert image.shape == (4567, 123, 1)


def test_generates_correct_bounding_boxes_for_rectangles():
    image, labels = generate_shapes(
        width=128,
        height=128,
        max_shapes=1,
        shape='rectangle')
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
        width=128,
        height=128,
        max_shapes=1,
        shape='triangle')
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
        width=43,
        height=44,
        max_shapes=1,
        min_dimension=20,
        max_dimension=20,
        shape='circle')
    assert len(labels) == 1
    label = labels[0]
    crop = image[label.y1:label.y2, label.x1:label.x2]

    # The crop is filled.
    assert (crop < 255).any()

    # The crop is complete.
    image[label.y1:label.y2, label.x1:label.x2] = 255
    assert (image == 255).all()


def test_generate_circle_throws_when_dimension_too_small():
    with pytest.raises(ValueError):
        generate_shapes(
            width=64,
            height=128,
            max_shapes=1,
            min_dimension=1,
            max_dimension=1,
            shape='circle')


def test_generate_triangle_throws_when_dimension_too_small():
    with pytest.raises(ValueError):
        generate_shapes(
            width=128,
            height=64,
            max_shapes=1,
            min_dimension=1,
            max_dimension=1,
            shape='triangle')


def test_can_generate_one_by_one_rectangle():
    image, labels = generate_shapes(
        width=50,
        height=128,
        max_shapes=1,
        min_dimension=1,
        max_dimension=1,
        shape='rectangle')
    assert len(labels) == 1
    label = labels[0]
    crop = image[label.y1:label.y2, label.x1:label.x2]
    assert (crop < 255).sum() == 3  # rgb


def test_throws_when_min_intensity_out_of_range():
    with pytest.raises(ValueError):
        generate_shapes(
            width=1000,
            height=1234,
            max_shapes=1,
            min_intensity=256)
    with pytest.raises(ValueError):
        generate_shapes(
            width=2,
            height=2,
            max_shapes=1,
            min_intensity=-1)


def test_returns_empty_labels_and_white_image_when_cannot_fit_shape():
    # The circle will never fit this.
    image, labels = generate_shapes(
        width=10000,
        height=10000,
        max_shapes=1,
        min_dimension=10000,
        shape='circle')
    assert len(labels) == 0
    assert (image == 255).all()
