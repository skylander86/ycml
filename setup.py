import os
from setuptools import setup, find_packages
import sys

if sys.version_info.major < 3:
    raise Exception('This is a Python 3 only package. Please upgrade.')
#end if

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(SCRIPT_DIR, 'README.rst'), 'r') as readme_file:
    readme = readme_file.read()

with open(os.path.join(SCRIPT_DIR, 'requirements.txt'), 'r') as f:
    requirements = list(filter(None, (line.strip() for line in f if not line.startswith('#'))))

with open(os.path.join(SCRIPT_DIR, 'VERSION'), 'r') as f:
    version = f.read().strip()

setup(
    name='ycml',
    version=version,
    author='yc sim',
    author_email='hello@yanchuan.sg',
    description='yc\'s collection of convenience code for developing ML applications.',
    long_description=readme,
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3 :: Only',
        'Natural Language :: English',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries',
        'Topic :: Utilities',
        'Operating System :: POSIX',
    ],
    keywords=['machine learning', 'ml', 'natural language processing', 'nlp', 'utilities'],
    url='http://github.com/skylander86/ycml',
    license='Apache Software License 2.0',
    packages=find_packages('.'),
    install_requires=requirements,
    include_package_data=True,
    zip_safe=False,
)
