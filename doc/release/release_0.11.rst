Announcement: scikit-image 0.11.0
=================================

We're happy to announce the release of scikit-image v0.11.0!

scikit-image is an image processing toolbox for SciPy that includes algorithms
for segmentation, geometric transformations, color space manipulation,
analysis, filtering, morphology, feature detection, and more.

For more information, examples, and documentation, please visit our website:

http://scikit-image.org

Highlights
----------
For this release, we merged over 200 pull requests with bug fixes,
cleanups, improved documentation and new features.  Highlights
include:

- Region Adjacency Graphs
  - Color distance RAGs (#1031)
  - Threshold Cut on RAGs (#1031)
  - Similarity RAGs (#1080)
  - Normalized Cut on RAGs (#1080)
  - RAG drawing (#1087)
  - Hierarchical merging (#1100)
- Sub-pixel shift registration (#1066)
- Non-local means denoising (#874)
- Sliding window histogram (#1127)
- More illuminants in color conversion (#1130)
- Handling of CMYK images (#1360)
- `stop_probability` for RANSAC (#1176)
- Li thresholding (#1376)
- Signed edge operators (#1240)
- Full ndarray support for `peak_local_max` (#1355)
- Improve conditioning of geometric transformations (#1319)
- Standardize handling of multi-image files (#1200)
- Ellipse structuring element (#1298)
- Multi-line drawing tool (#1065), line handle style (#1179)
- Point in polygon testing (#1123)
- Rotation around a specified center (#1168)
- Add `shape` option to drawing functions (#1222)
- Faster regionprops (#1351)
- `skimage.future` package (#1365)
- More robust I/O module (#1189)

API Changes
-----------
- The ``skimage.filter`` subpackage has been renamed to ``skimage.filters``.
- Some edge detectors returned values greater than 1--their results are now
  appropriately scaled with a factor of ``sqrt(2)``.

Contributors to this release
----------------------------
(Listed alphabetically by last name)

- Fedor Baart
- Vighnesh Birodkar
- François Boulogne
- Nelson Brown
- Alexey Buzmakov
- Julien Coste
- Phil Elson
- Adam Feuer
- Jim Fienup
- Geoffrey French
- Emmanuelle Gouillart
- Charles Harris
- Jonathan Helmus
- Alexander Iacchetta
- Ivana Kajić
- Kevin Keraudren
- Almar Klein
- Gregory R. Lee
- Jeremy Metz
- Stuart Mumford
- Damian Nadales
- Pablo Márquez Neila
- Juan Nunez-Iglesias
- Rebecca Roisin
- Jasper St. Pierre
- Jacopo Sabbatini
- Michael Sarahan
- Salvatore Scaramuzzino
- Phil Schaf
- Johannes Schönberger
- Tim Seifert
- Arve Seljebu
- Steven Silvester
- Julian Taylor
- Matěj Týč
- Alexey Umnov
- Pratap Vardhan
- Stefan van der Walt
- Joshua Warner
- Tony S Yu
