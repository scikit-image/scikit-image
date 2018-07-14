CellProfiler BSD license announcement
-------------------------------------

::

  From: Vebjorn Ljosa
  To: scikit-image@googlegroups.com
  Date: June 3, 2010

  We have changed the license of some parts of CellProfiler from GNU GPL
  to BSD.  It has previously been proposed [1] that some of the image
  processing code in CellProfiler be merged into SciPy, and the license
  change makes this possible.

  The CellProfiler SVN repository is at
  https://svn.broadinstitute.org/CellProfiler/trunk/CellProfiler/.  The
  file LICENSE [2] contains a list of BSD-licensed subdirectories as
  well as other license details.  The rest of CellProfiler continues to
  be licensed under the GNU GPL.  The BSD-licensed subdirectories are:

   * CellProfiler/cpmath [3]: image processing algorithms
   * CellProfiler/utilities [4]: contains a Java bridge, making it
  possible to call Java functions from Python
   * bioformats [5]: wrapper that uses the Java bridge to have
  Bioformats [6] read or write an image file

  Good luck with the upcoming scikits.image sprint.  I don't think
  anyone from the CellProfiler team will be able to take part in the
  sprint this time, but don't hesitate to ask on the
  cellprofiler-dev@broadinstitute.org mailing list.

  Thanks,
  Vebjorn

  [1] http://stefanv.github.com/scikits.image/contribute.html
  [2] https://svn.broadinstitute.org/CellProfiler/trunk/CellProfiler/LICENSE
  [3] https://svn.broadinstitute.org/CellProfiler/trunk/CellProfiler/cellprofiler/cpmath/
  [4] https://svn.broadinstitute.org/CellProfiler/trunk/CellProfiler/cellprofiler/utilities/
  [5] https://svn.broadinstitute.org/CellProfiler/trunk/CellProfiler/bioformats/
  [6] http://www.loci.wisc.edu/software/bio-formats
