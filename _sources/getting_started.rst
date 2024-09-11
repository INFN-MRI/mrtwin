Getting Started
===============

Installing MRTWIN
-----------------

MRTwin is available on PyPi

.. code-block:: sh

    pip install mrtwin

Development Version
~~~~~~~~~~~~~~~~~~~

If you want to modifiy the mrtwin code base

.. code-block:: sh

    git clone https://github.com/INFN-MRI/mrtwin
    pip install -e ./mrtwin[test, dev, doc]


Basic Usage
===========

Using MRTwin, we can quickly create a Shepp-Logan phantom,
the corresponding static field inhomogeneity map and a set 
of coil sensitivity maps as follows

.. code-block:: python

    import mrtwin

    # 2D Shepp-Logan phantom
    phantom = mrtwin.shepplogan_phantom(ndim=2, shape=256).as_numeric()

    # B0 map
    b0_map = mrtwin.b0field(phantom.Chi)

    # Coil sensitivity maps
    smaps = mrtwin.sensmap(shape=(8, 256, 256))

This allow us to quickly simulate, e.g., a fully-sampled multi-coil Cartesian GRE experiment
as:

.. code-block:: python

    import numpy as np 

    TE = 10.0 # ms
    rate_map = phantom.T2s + 1j * 2 * np.pi * b0_map
    gre = smaps * phantom.M0 * np.exp(-rate_map * TE)

This can be coupled with other libraries (e.g., `MRI-NUFFT <https://github.com/mind-inria/mri-nufft>`_)
to simulate more complex MR sequences (e.g., Non-Cartesian and sub-Nyquist imaging).

