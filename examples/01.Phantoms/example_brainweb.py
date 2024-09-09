"""
================
Brainweb Phantom
================

Example of Brainweb phantom creation.

This examples show how to generate numerical phantoms based on the Brainweb
dataset.

"""

import matplotlib.pyplot as plt
import numpy as np

from mrtwin import brainweb_phantom

plt.rcParams["image.cmap"] = "gray"


# %%
# Basic Usage
# ===========
# The Brainweb database consists of a set of 20 normal brains
# (ids equal to `[4, 5, 6, 18, 20, 38, 41-54]) at a nominal 0.5 mm isotropic
# resolution and a shape of `(nz, ny, nx) = (362, 434, 362)`.
#
# A digital Brainweb phantom can be created as:

phantom2D = brainweb_phantom(
    ndim=2, subject=4, segtype="fuzzy"
)  # single-slice 2D phantom
phantom3D = brainweb_phantom(ndim=3, subject=4, segtype="fuzzy")  # 3D phantom

# %%
# The phantoms here created are sparse, i.e., they consists of a
# `(nclasses, *spatial_shape)` shaped `np.ndarray` representing the
# probabilistic maps of each tissue type (e.g., Gray Matter, White Matter, CSF)
# and a list of `(nclasses,)` dictionaries each containing the `(M0, T1, T2, T2*, Chi)`
# values for each class:

# Print summary

print(phantom2D)
print(phantom3D)

# Display spatial segmentations

example2D = np.concatenate((phantom2D[3], phantom2D[2], phantom2D[1]), axis=0)

example3Dax = np.concatenate(
    (phantom3D[3, 100], phantom3D[2, 100], phantom3D[1, 100]), axis=0
)
example3Dcor = np.concatenate(
    (phantom3D[3, ::-1, 100], phantom3D[2, ::-1, 100], phantom3D[1, ::-1, 100]), axis=0
)
example3Dsag = np.concatenate(
    (
        phantom3D[3, ::-1, :, 100],
        phantom3D[2, ::-1, :, 100],
        phantom3D[1, ::-1, :, 100],
    ),
    axis=0,
)
example3D = np.concatenate((example3Dax, example3Dcor, example3Dsag), axis=1)

# %%
# The `(M0, T1, T2, T2*, Chi)` properties
# can be direcly accessed as:

_, _ = print("M0:", end="\t"), print(phantom2D.M0)  # same for phantom3D
_, _ = print("T1 (ms):", end="\t"), print(phantom2D.T1)
_, _ = print("T2 (ms):", end="\t"), print(phantom2D.T2)
_, _ = print("T2* (ms):", end="\t"), print(phantom2D.T2s)
_, _ = print("Chi (ppb):", end="\t"), print(phantom2D.Chi)


# %%
# If required, the '`properties` dictionary
# can be directly accessed as

print(phantom2D.properties)

# %%
# e.g., to be passed as `**kwargs` to a simulator routine.
#
# Notice that a basic summary of the properties can be accessed
# via the `__repr__` attribute (i.e., enabling pretty printing), while
# segmentation can be accessed directly (in read-only mode)
# via square bracked indexing, similarly to numpy arrays.

fig1, ax1 = plt.subplots(1, 2)
ax1[0].imshow(example2D), ax1[0].axis("off"), ax1[0].set_title("2D phantom")
ax1[1].imshow(example3D), ax1[1].axis("off"), ax1[1].set_title("3D phantom")
plt.show()

# %%
# We also provide a crisp segmentation, which has a lower memory footprint
# at cost of a coarser approximation (i.e., a piecewise-constant tissue model)-
#
# This can be obtained starting from the `"fuzzy"` phantom as:

phantom2D = phantom2D.as_crisp()
phantom3D = phantom3D.as_crisp()

# Print summary

print(phantom2D)
print(phantom3D)

# Display spatial segmentations

example2D = phantom2D

example3Dax = np.concatenate((phantom3D[100], phantom3D[100], phantom3D[100]), axis=0)
example3Dcor = np.concatenate(
    (phantom3D[::-1, 100], phantom3D[::-1, 100], phantom3D[::-1, 100]), axis=0
)
example3Dsag = np.concatenate(
    (phantom3D[::-1, :, 100], phantom3D[::-1, :, 100], phantom3D[::-1, :, 100]), axis=0
)
example3D = np.concatenate((example3Dax, example3Dcor, example3Dsag), axis=1)

fig2, ax2 = plt.subplots(1, 2)
ax2[0].imshow(example2D, cmap="turbo"), ax2[0].axis("off"), ax2[0].set_title(
    "2D phantom"
)
ax2[1].imshow(example3D, cmap="turbo"), ax2[1].axis("off"), ax2[1].set_title(
    "3D phantom"
)
plt.show()

# %%
# Crisp phantom can be also directly generated as:

phantom2D = brainweb_phantom(
    ndim=2, subject=4, segtype="crisp"
)  # single-slice 2D phantom
phantom3D = brainweb_phantom(ndim=3, subject=4, segtype="crisp")  # 3D phantom

# N.B. 'segtype' can be omitted as the default is `"crisp"`.

phantom2D = brainweb_phantom(ndim=2, subject=4)  # single-slice 2D phantom
phantom3D = brainweb_phantom(ndim=3, subject=4)  # 3D phantom

# %%
# Finally, we can obtain a "dense" phantom,
# i.e., an object without segmentation whose
# `(M0, T1, T2, T2*, Chi)` properties are stored
# as parametric maps rather than the individual values
# of each tissue class.
#
# This can be obtain (both from `"fuzzy"` and `"crisp"` models) as:

phantom2D = phantom2D.as_numeric()

# Print summary

print(phantom2D)

# Display spatial segmentations

fig3, ax3 = plt.subplots(1, 5)

im0 = ax3[0].imshow(phantom2D.M0, cmap="gray")
ax3[0].axis("off"), ax3[0].set_title("M0 [a.u.]")
fig3.colorbar(im0, ax=ax3[0], fraction=0.046, pad=0.04)

im1 = ax3[1].imshow(phantom2D.T1, cmap="magma")
ax3[1].axis("off"), ax3[1].set_title("T1 [ms]")
fig3.colorbar(im1, ax=ax3[1], fraction=0.046, pad=0.04)

im2 = ax3[2].imshow(phantom2D.T2, cmap="viridis", vmax=150)
ax3[2].axis("off"), ax3[2].set_title("T2 [ms]")
fig3.colorbar(im2, ax=ax3[2], fraction=0.046, pad=0.04)

im2 = ax3[3].imshow(phantom2D.T2s, cmap="viridis", vmax=150)
ax3[3].axis("off"), ax3[3].set_title("T2* [ms]")
fig3.colorbar(im2, ax=ax3[3], fraction=0.046, pad=0.04)

im4 = ax3[4].imshow(phantom2D.Chi, cmap="gray")
ax3[4].axis("off"), ax3[4].set_title("Chi [ppb]")
fig3.colorbar(im4, ax=ax3[4], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()

# %%
# Hereafter, without loss of generality, we will use 2D phantoms.
#
# Dense phantom can be also directly generated as:

phantom2D = brainweb_phantom(
    ndim=2, subject=4, segtype=False
)  # single-slice 2D phantom

# Print summary

print(phantom2D)

# %% Setting spatial properties
#
# By default, Brainweb phantoms are interpolated
# to 1.085 mm isotropic resolution with a 200 isotropic matrix (217 mm iso FOV).
#
# These can be adjusted using `shape` and `output_res` model:
#
# 1. `shape` will control the matrix size without affecting the resolution.
# 2. `output_res` will adjust the spatial resolution keeping the same (i.e., 200 iso) matrix
#

phantom2D = brainweb_phantom(ndim=2, subject=4)
phantom2D_mtx = brainweb_phantom(
    ndim=2, subject=4, shape=256
)  # can also be shape=(ny, nx)
phantom2D_res = brainweb_phantom(
    ndim=2, subject=4, output_res=2.0
)  # can also be output_res=(dy, dx)
phantom2D_mtx_res = brainweb_phantom(ndim=2, subject=4, shape=256, output_res=2.0)

print(phantom2D)
print(phantom2D_mtx)
print(phantom2D_res)
print(phantom2D_mtx_res)

fig4, ax4 = plt.subplots(2, 2)
ax4[0, 0].imshow(phantom2D, cmap="turbo"), ax4[0, 0].axis("off"), ax4[0, 0].set_title(
    "shape 200, res=1.085mm, fov=217mm"
)
ax4[0, 1].imshow(phantom2D_mtx, cmap="turbo"), ax4[0, 1].axis("off"), ax4[
    0, 1
].set_title("shape 256, res=1.085mm, fov=277mm")
ax4[1, 0].imshow(phantom2D_res, cmap="turbo"), ax4[1, 0].axis("off"), ax4[
    1, 0
].set_title("shape 200, res=2.0mm, fov=400mm")
ax4[1, 1].imshow(phantom2D_mtx_res, cmap="turbo"), ax4[1, 1].axis("off"), ax4[
    1, 1
].set_title("shape 256, res=2.0mm, fov=512mm")
plt.show()


# %% Setting field strength
#
