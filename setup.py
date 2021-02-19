#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='ecg_uncertainty',
    version='0.0.1',
    description='Repository accompanying the paper "Uncertainty estimation for deep learning-based automated analysis of 12-lead electrocardiograms" by Vranken and Van de Leur et al. It contains all the uncertainty estimation methods described in the paper.',
    author='Jeroen F. Vranken',
    author_email='jeroen.vranken@xs4all.nl',
    # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
    url='https://github.com/rutgervandeleur/uncertainty',
    install_requires=['pytorch-lightning'],
    packages=find_packages(),
)

