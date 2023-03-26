"""
===============================================
Reconstruct dust-covered image using inpainting
===============================================

It is possible that dust gets accumulated on the reference mirror and causes
dark spots to appear on direct images. This example reproduces the steps taken
to perform OCT (Optical Coherence Tomography [1]_) dust removal in images.
This application was first discussed by Jules Scholler in [2]_.

.. [1] Vinay A. Shah M.D. (2015)
       `Optical Coherence Tomography <https://eyewiki.aao.org/Optical_Coherence_Tomography#:~:text=3%20Limitations-,Overview,at%20least%2010%2D15%20microns.>`_,
       American Academy of Ophthalmology.
.. [2] Jules Scholler (2019) "Image denoising using inpainting":
       `<https://www.jscholler.com/2019-02-28-remove-dots/>`_

"""
