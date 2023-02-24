"""
This module is an example of a barebones numpy reader plugin for napari.

It implements the Reader specification, but your plugin may choose to
implement multiple readers or even other plugin contributions. see:
https://napari.org/stable/plugins/guides.html?#readers
"""
import cv2
import numpy as np
import requests
from pynwb import NWBHDF5IO
from pynwb.image import ImageSeries


def napari_get_reader(path):
    """A basic implementation of a Reader contribution.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    function or None
        If the path is a recognized format, return a function that accepts the
        same path or list of paths, and returns a list of layer data tuples.
    """
    if isinstance(path, list):
        # reader plugins may be handed single path, or a list of paths.
        # if it is a list, it is assumed to be an image stack...
        # so we are only going to look at the first file.
        path = path[0]

    # if we know we cannot read the file, we immediately return None.
    if not path.endswith(".nwb"):
        return None

    # otherwise we return the *function* that can read ``path``.
    return reader_function


def reader_function(path):
    """Take a path or list of paths and return a list of LayerData tuples.

    Readers are expected to return data as a list of tuples, where each tuple
    is (data, [add_kwargs, [layer_type]]), "add_kwargs" and "layer_type" are
    both optional.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    layer_data : list of tuples
        A list of LayerData tuples where each tuple in the list contains
        (data, metadata, layer_type), where data is a numpy array, metadata is
        a dict of keyword arguments for the corresponding viewer.add_* method
        in napari, and layer_type is a lower-case string naming the type of
        layer. Both "meta", and "layer_type" are optional. napari will
        default to layer_type=="image" if not provided
    """
    nwb_io = NWBHDF5IO(path, mode="r")
    nwb_file = nwb_io.read()
    image_series = nwb_file.acquisition["image_series"]
    data = _read_external_jpg_imageseries(image_series)

    # optional kwargs for the corresponding viewer.add_* method
    add_kwargs = {}

    layer_type = "image"  # optional, default is "image"
    return [(data, add_kwargs, layer_type)]


def _read_external_jpg_slice(url: str):
    buffer = requests.get(url).content
    bytes = np.frombuffer(buffer, np.uint8)
    slice = cv2.imdecode(bytes, cv2.IMREAD_GRAYSCALE)
    return slice


def _read_external_jpg_imageseries(image_series: ImageSeries):
    assert (
        image_series.external_file is not None
    ), "Trying to read NWB ImageSeries' external data, but there is none."
    image_stack = None
    for slice_index in range(len(image_series.external_file)):
        slice = _read_external_jpg_slice(
            image_series.external_file[slice_index]
        )
        if slice_index == 0:
            image_stack = np.array(
                [slice]
            )  # add extra dimension here, so vstack will work later
        else:
            image_stack = np.vstack((image_stack, np.array([slice])))
    print(image_stack.shape)
    return image_stack
