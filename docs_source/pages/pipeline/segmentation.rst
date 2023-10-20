Segmentation
============

Introduction
------------
This workflow performs cell segmentation using respective Cellpose models. It segments both the nucleus and the cytosol of cells in an input image based on filtering threshold set by the user.

Requirements
------------
- NumPy
- PyTorch
- Cellpose
- Python's garbage collection (`gc`)
- Matplotlib

Usage
-----
.. code-block:: python

    # add usage code here

Workflow
--------

1. **Memory Cleanup**: 
   - Invokes Python's garbage collection using `gc.collect()`.
   - Clears PyTorch's GPU cache using `torch.cuda.empty_cache()`.

2. **Image Preprocessing**: 
   - Converts the input image to a 16-bit unsigned integer using `input_image.astype(np.uint16)`.

3. **GPU Check**: 
   - Uses `torch.cuda.is_available()` to check for GPU availability.
   - Sets the `use_GPU` flag accordingly.

4. **Model Loading**: 
   - Reads the configuration to determine which Cellpose models to use for nucleus and cytosol segmentation.
   - Calls `_read_cellpose_model()` to load the appropriate model based on the configuration.

5. **Diameter Configuration**: 
   - Checks if a specific diameter is set in the configuration for both nucleus and cytosol segmentation.
   - If not, defaults to `None`.

6. **Segmentation**: 
   - Calls `model.eval()` to perform the segmentation.
   - The segmented masks are then converted to NumPy arrays.

7. **Mask Filtering**: 
   - Iterates through all unique nucleus IDs in the nucleus mask.
   - For each nucleus ID, finds the corresponding cytosol IDs in the cytosol mask. If the cytosol mask is empty, the nucleus ID is skipped.
   - Utilizes `self.config["filtering_threshold"]` to determine the minimum proportion of cytosol ID that must correspond to a nucleus for it to be considered a valid pair. 
   If the proportion is less than the threshold, the nucleus ID is skipped.
   - Checks if any cytosol regions are assigned to multiple nuclei or empty and reassigns them to background.

8. **Final Mask Updates**: 
    - The nucleus and cytosol IDs are then paired together and added to a dictionary `nucleus_cytosol_pairs`.
    - Saves the final masks to the `self.maps` attribute.

9. **Debugging**: 
    - If in debug mode, visualizes the masks before and after filtering using Matplotlib.

10. **Performance Logging**: 
    - Logs the time required for the mask filtering process.

11. **Memory Cleanup**: 
    - Deletes temporary variables and performs garbage collection and GPU cache clearing one final time.

Examples
--------

.. code-block:: python

    # add examples?

Figures
-------
.. image:: /docs/html/_images/segmentation_example.png
   :alt: Before and after cell segmentation with 0.95 filtering threshold

FAQs
----
1. **Why is my GPU memory not freeing up?**
   - Make sure to call `gc.collect()` and `torch.cuda.empty_cache()`.

2. **Why are my masks not accurate?**
   - Check the `filtering_threshold` parameter in the configuration. Also check the model and diameter parameters.

References
----------
- Cellpose GitHub repository: `Cellpose GitHub <https://github.com/MouseLand/cellpose>`_
- PyTorch Documentation: `PyTorch Docs <https://pytorch.org/docs/stable/index.html>`_

