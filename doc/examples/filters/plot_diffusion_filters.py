"""
=================
Diffusion filters
=================
The diffusion process can be seen as an evolution process with an artificial
time variable t ( t = time_step * num_iters) denoting the diffusion time.
The bigger the time_step is, the lower the num_iters parameter has to be
and the faster the computation is. However, for explicit scheme the maximal
stable value of time_step is 0.25.

In this example the difference between implemented diffusion filters is shown.
AOS scheme is used, as this scheme is more time efficient.

Linear diffusion
-----------------
The input image is smoothed at a constant rate in all directions.
In theory, it corresponds to a Gaussian filter with sigma = sqrt(2 * t).

Nonlinear isotropic diffusion
------------------------------
Diffusivity is a scalar set according to the chosen diffusivity
type (Perona-Malik, Charbonnier, exponential). Diffusivity is set
for each pixel individually and the diffusion flux is the same for every
direction. Diffusivity is smaller near edges.


Nonlinear anisotropic diffusion - Edge Enhancing Diffusion (EED)
----------------------------------------------------------------
A diffusion tensor is present that indicates the directions of filtering.
This filter preserves edges while smoothing homogeneous structures.

Nonlinear anisotropic diffusion - Coherence Enhancing Diffusion (CED)
---------------------------------------------------------------------
A diffusion tensor is present that indicates the directions of filtering.
This filter smoothes the image along edges and minimizes smoothing
across edges. The output image has enhanced coherent, flow-like structures.

"""
import matplotlib.pyplot as plt
from skimage.filters._diffusion_nonlinear_aniso import diffusion_nonlinear_aniso
from skimage.filters._diffusion_linear import diffusion_linear
from skimage.filters._diffusion_nonlinear_iso import diffusion_nonlinear_iso
from skimage.util import random_noise
from skimage import data

aneurysms = data.microaneurysms()
astronaut = data.astronaut()[15:190, 150:300]

"""
Diffusion on noisy images. Image data.microaneurysms() is chosen
to show advantages and results of nonlinear anisotropic
diffusion (CED). Parameters of each diffusion are different
and are set to show better result of respective diffusion.
"""
noisy = random_noise(aneurysms, var=0.0005)

fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(8, 5),
                       sharex=True, sharey=True)
plt.gray()
ax[0, 0].imshow(aneurysms)
ax[0, 0].axis('off')
ax[0, 0].set_title('aneurysms')
ax[0, 1].imshow(diffusion_linear(noisy, time_step=2, num_iters=4,
                                 scheme='aos'))
ax[0, 1].axis('off')
ax[0, 1].set_title('Linear diffusion')
ax[0, 2].imshow(diffusion_nonlinear_aniso(noisy, mode='eed', time_step=2,
                                          num_iters=5, scheme='aos',
                                          sigma_eed=2.5))
ax[0, 2].axis('off')
ax[0, 2].set_title('Nonlinear anisotropic diffusion (EED)')
ax[0, 3].imshow(diffusion_nonlinear_aniso(noisy, mode='ced', time_step=2,
                                          num_iters=10, scheme='aos',
                                          sigma_ced=0.5, rho=8., lmbd=2.))
ax[0, 3].axis('off')
ax[0, 3].set_title('Nonlinear anisotropic diffusion (CED)')

ax[1, 1].imshow(diffusion_nonlinear_iso(noisy, diffusivity_type='charbonnier',
                                        time_step=2, num_iters=10,
                                        scheme='aos', sigma=1.5, lmbd=2.))
ax[1, 1].axis('off')
ax[1, 1].set_title('Nonlinear isotropic diffusion (Charbonnier)')
ax[1, 2].imshow(diffusion_nonlinear_iso(noisy, diffusivity_type='exponential',
                                        time_step=2, num_iters=10,
                                        scheme='aos', sigma=2.5, lmbd=2.))
ax[1, 2].axis('off')
ax[1, 2].set_title('Nonlinear isotropic diffusion (exponential)')
ax[1, 3].imshow(diffusion_nonlinear_iso(noisy, diffusivity_type='perona-malik',
                                        time_step=2, num_iters=5, scheme='aos',
                                        sigma=1.5, lmbd=2.))
ax[1, 3].axis('off')
ax[1, 3].set_title('Nonlinear isotropic diffusion (Perona-Malik)')
ax[1, 0].imshow(noisy)
ax[1, 0].axis('off')
ax[1, 0].set_title('noisy')
fig.tight_layout()
plt.show()


"""
Diffusion on color images with default parameters.
"""
noisy = random_noise(astronaut, var=0.005)

fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(8, 5),
                       sharex=True, sharey=True)
plt.gray()
ax[0, 0].imshow(astronaut)
ax[0, 0].axis('off')
ax[0, 0].set_title('astronaut')
ax[0, 1].imshow(diffusion_linear(noisy))
ax[0, 1].axis('off')
ax[0, 1].set_title('Linear diffusion')
ax[0, 2].imshow(diffusion_nonlinear_aniso(noisy))
ax[0, 2].axis('off')
ax[0, 2].set_title('Nonlinear anisotropic diffusion (EED)')
ax[0, 3].imshow(diffusion_nonlinear_aniso(noisy, mode='ced'))
ax[0, 3].axis('off')
ax[0, 3].set_title('Nonlinear anisotropic diffusion (CED)')

ax[1, 1].imshow(diffusion_nonlinear_iso(noisy, diffusivity_type='charbonnier'))
ax[1, 1].axis('off')
ax[1, 1].set_title('Nonlinear isotropic diffusion (Charbonnier)')
ax[1, 2].imshow(diffusion_nonlinear_iso(noisy, diffusivity_type='exponential'))
ax[1, 2].axis('off')
ax[1, 2].set_title('Nonlinear isotropic diffusion (exponential)')
ax[1, 3].imshow(diffusion_nonlinear_iso(noisy))
ax[1, 3].axis('off')
ax[1, 3].set_title('Nonlinear isotropic diffusion (Perona-Malik)')
ax[1, 0].imshow(noisy)
ax[1, 0].axis('off')
ax[1, 0].set_title('Noisy')
fig.tight_layout()
plt.show()
