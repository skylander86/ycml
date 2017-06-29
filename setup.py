from setuptools import setup, find_packages

setup(
    name='ycml',
    version='0.1',
    author='yc sim',
    author_email='hello@yanchuan.sg',
    description='yc\'s collection of convenience code for developing ML applications.',
    long_description='This module contains a collection of code I\'ve written over the past years for pretty run-of-the-mill deployments of machine learning projects.',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
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
    install_requires=[
        'matplotlib>=2.0',
        'numpy>=1.12',
        'scipy>=0.19',
        'scikit-learn>=0.18',
        'tabulate',
        'pyyaml',
    ],
    include_package_data=True,
    zip_safe=False,
)
