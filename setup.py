from setuptools import setup, find_packages
import sys

if sys.version_info.major < 3:
    raise Exception('This is a Python 3 only package. Please upgrade.')
#end if

setup(
    name='ycml',
    version='0.1.9',
    author='yc sim',
    author_email='hello@yanchuan.sg',
    description='yc\'s collection of convenience code for developing ML applications.',
    long_description=open('README.md', 'r').read(),
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
    install_requires=[line.strip() for line in open('requirements.txt', 'r') if not line.startswith('#') and line.strip()],
    include_package_data=True,
    zip_safe=False,
)
