L1:# -*- coding: utf-8 -*-
L2+# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
L3+# vi: set ft=python sts=4 ts=4 sw=4 et:
L4+"""
L5+Base module variables for ModelArrayIO
L6+"""
L7+
L8+from .__about__ import __version__
L9+
L10+__author__ = 'The PennLINC Developers'
L11+__copyright__ = 'Copyright 2021, PennLINC, Perelman School of Medicine, University of Pennsylvania'
L12+__credits__ = ['Matt Cieslak', 'Tinashe Tapera', 'Chenying Zhao',
L13+               'Steven Meisler', 'Valerie Sydnor', 'Josiane Bourque']
L14+__license__ = '3-clause BSD'
L15+__maintainer__ = 'Matt Cieslak'
L16+__status__ = 'Prototype'
L17+__url__ = 'https://github.com/PennLINC/ModelArrayIO'
L18+__packagename__ = 'modelarrayio'
L19+__description__ = "ModelArrayIO converters for fixel/voxel/greyordinate data"
L20+__longdesc__ = """\
L21+A package that converts between imaging formats and the HDF5 file format used by ModelArray.
L22+"""
L23+
L24+DOWNLOAD_URL = (
L25+    'https://github.com/pennlinc/{name}/archive/{ver}.tar.gz'.format(
L26+        name=__packagename__, ver=__version__))
L27+
L28+
L29+SETUP_REQUIRES = [
L30+    'setuptools>=18.0',
L31+    'numpy',
L32+    'cython',
L33+]
L34+
L35+REQUIRES = [
L36+    'numpy',
L37+    'future',
L38+    'nilearn',
L39+    'nibabel>=2.2.1',
L40+    'pandas',
L41+    'h5py',
L42+]
L43+
L44+LINKS_REQUIRES = [
L45+]
L46+
L47+TESTS_REQUIRES = [
L48+    "mock",
L49+    "codecov",
L50+    "pytest",
L51+]
L52+
L53+EXTRA_REQUIRES = {
L54+    'doc': [
L55+        'sphinx>=1.5.3',
L56+        'sphinx_rtd_theme',
L57+        'sphinx-argparse',
L58+        'pydotplus',
L59+        'pydot>=1.2.3',
L60+        'packaging',
L61+        'nbsphinx',
L62+    ],
L63+    'tests': TESTS_REQUIRES,
L64+    'duecredit': ['duecredit'],
L65+    'datalad': ['datalad'],
L66+    'resmon': ['psutil>=5.4.0'],
L67+}
L68+EXTRA_REQUIRES['docs'] = EXTRA_REQUIRES['doc']
L69+
L70+# Enable a handle to install all extra dependencies at once
L71+EXTRA_REQUIRES['all'] = list(set([
L72+    v for deps in EXTRA_REQUIRES.values() for v in deps]))
L73+
L74+CLASSIFIERS = [
L75+    'Development Status :: 3 - Alpha',
L76+    'Intended Audience :: Science/Research',
L77+    'Topic :: Scientific/Engineering :: Image Recognition',
L78+    'License :: OSI Approved :: BSD License',
L79+    'Programming Language :: Python :: 3.5',
L80+    'Programming Language :: Python :: 3.6',
L81+    'Programming Language :: Python :: 3.7',
L82+]
L83+

