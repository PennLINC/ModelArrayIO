Installation
============

ModelArrayIO can be installed from pip. To install the latest official release:

.. code-block:: bash

    pip install modelarrayio

If you want to use the most up-to-date version, you can install from the ``main`` branch:

.. code-block:: bash

    pip install git+https://github.com/PennLINC/ModelArrayIO.git


MRtrix (required for fixel ``.mif`` only)
-----------------------------------------

For fixel-wise ``.mif`` conversion, the ``modelarrayio mif-to-h5`` / ``modelarrayio h5-to-mif`` tools use MRtrix ``mrconvert``.
Install MRtrix from `MRtrix's webpage <https://www.mrtrix.org/download/>`_ if needed.
Run ``mrview`` in the terminal to verify the installation.

If your data are in NIfTI or CIFTI format only, you can skip this step.


What Next?
----------

For an overview of what you can do with ModelArrayIO see the `ModelArrayIO documentation <https://modelarrayio.readthedocs.io/en/latest/>`_.

For an overview of what you can do with ModelArray see the `ModelArray documentation <https://pennlinc.github.io/ModelArray/>`_.

To get right to using ModelArrayIO see the documentation on the `command line interface <https://modelarrayio.readthedocs.io/en/latest/usage.html>`_.

If you have questions, or need help with using ModelArrayIO or ModelArray, check out `NeuroStars <https://neurostars.org/tags/c/software-support/234/modelarray>`_.
