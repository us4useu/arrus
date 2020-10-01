.. _arrus-api-main:

=============
API Reference
=============

.. caution::

    We will do our best to maintain backward compatibility but please note
    that ARRUS is currently under development and its API may be modified
    in the future.

.. _arrus-Us4R:

Handle to the Us4R System
=====================

.. mat:autoclass:: arrus.Us4R
    :show-inheritance:
    :members:

Operations
==========

Each operation derives from the Operation class.

.. _arrus.Operation:

.. mat:autoclass:: arrus.Operation
    :show-inheritance:
    :members:

.. _arrus-api-sequences:

TX/RX sequences
~~~~~~~~~~~~~~~

All TX/RX sequences derive parameters from the :class:`SimpleTxRxSequence` class.

.. _arrus.SimpleTxRxSequence:

.. mat:autoclass:: arrus.SimpleTxRxSequence
    :show-inheritance:
    :members:


Specific sequences currently available in the system are as follows:

.. _arrus.PWISequence:

.. mat:autoclass:: arrus.PWISequence
    :show-inheritance:
    :members:

.. _arrus.STASequence:

.. mat:autoclass:: arrus.STASequence
    :show-inheritance:
    :members:
	
.. _arrus.LINSequence:

.. mat:autoclass:: arrus.LINSequence
    :show-inheritance:
    :members:


.. _arrus.Reconstruction:

B-mode Image Reconstruction
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. mat:autoclass:: arrus.Reconstruction
    :show-inheritance:
    :members:

Utilities
=========

.. _arrus.BModeDisplay:

.. mat:autoclass:: arrus.BModeDisplay
    :show-inheritance:
    :members:

