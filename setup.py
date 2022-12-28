import os
from setuptools import setup

with open(os.path.join(os.path.dirname(__file__), 'README.md')) as readme:
   README = readme.read()

os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))

setup(
    name='convRFF',
    version='1.0a0',
    packages=['convRFF'],

    author='Juan Carlos Aguirre Arango',
    author_email='jucaguirrear@unal.edu.co',
    maintainer='Juan Carlos Aguirre Arango',
    maintainer_email='jucaguirrear@unal.edu.co',

    download_url='',

    install_requires=[ 
                     'gdown==4.3.0',
                      'pandas',
                      'sklearn',
                      'tensorflow',
                      'tf-keras-vis @ git+https://github.com/UN-GCPDS/tf-keras-vis.git',
                      'gcpds.image_segmentation @ git+https://github.com/UN-GCPDS/python-gcpds.image_segmentation.git',
                      'scikit-image',
                      'matplotlib',
                      'tensorflow-datasets',
                      'tensorflow_addons',
                      'wandb'     
    ],

    include_package_data=True,
    license='MIT License',
    description="",
    zip_safe=False,

    long_description=README,
    long_description_content_type='text/markdown',

    python_requires='>=3.7',

)