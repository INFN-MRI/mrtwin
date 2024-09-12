MRTwin
======

MRTwin is a collection of virtual objects for numerical MR experiments.

.. list-table::
   :widths: 10 10 10 10 10 10 10 10 10
   :header-rows: 0

   * - .. image:: https://infn-mri.github.io/mrtwin/_static/coverage_badge.svg
     - .. image:: https://github.com/INFN-MRI/mrtwin/workflows/CI/badge.svg
     - .. image:: https://github.com/INFN-MRI/mrtwin/workflows/CD/badge.svg
     - .. image:: https://img.shields.io/github/license/INFN-MRI/mrtwin
        :target: https://github.com/INFN-MRI/mrtwin/blob/main/LICENSE.txt
     - .. image:: https://www.codefactor.io/repository/github/INFN-MRI/mrtwin/badge
        :target: https://www.codefactor.io/repository/github/INFN-MRI/mrtwin
     - .. image:: https://img.shields.io/badge/docs-Sphinx-blue
        :target: https://infn-mri.github.io/mrtwin
     - .. image:: https://img.shields.io/pypi/v/mrtwin
        :target: https://pypi.org/project/mrtwin
     - .. image:: https://img.shields.io/badge/style-black-black
     - .. image:: https://img.shields.io/badge/Python-%3E=3.9-blue?logo=python&logoColor=white
        :target: https://python.org
        :alt: Made with Python
        :title: Go to Python homepage

Features
--------

- **Virtual Phantoms:** A collection of sparse (fuzzy and crisp) and dense phantoms for quantitative MRI based on different anatomical models (Shepp-Logan, Brainweb database, Open Science CBS Neuroimaging Repository database) and different tissue representations (single pool, two- and three-pools).
- **Field Maps:** Routines for generation of realistic field maps, including B0 (based on input phantom susceptibility), B1 (including multiple RF modes) and coil sensitivities.
- **Motion patterns:** Markov chain generated rigid motion patterns (both for 2D and 3D imaging) to simulate the effect of motion on MR image quality.
- **Gradient System Response:** Generate Gaussian-shaped gradient response function with linear phase components to simulate k-space trajectory shift and deformation due to non-ideal gradient systems.

Installation
------------

MRTwin can be installed via pip as:

.. code-block:: bash

    pip install mrtwin

Basic Usage
-----------

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



Development
~~~~~~~~~~~

If you are interested in improving this project, install MRTwin in editable mode:

.. code-block:: bash

    git clone git@github.com:paquiteau/brainweb-dl 
    cd brainweb-dl
    pip install -e .[dev,test,doc]


Related projects
----------------

This package is inspired by the following excellent projects:

- Brainweb-dl <http://github.com/paquiteau/brainweb-dl>
- Phantominator <https://github.com/mckib2/phantominator>
- SigPy <https://github.com/mikgroup/sigpy>

