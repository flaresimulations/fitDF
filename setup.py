#!/usr/bin/env python

from distutils.core import setup
##from setuptools import setup

setup(name='fitDF',
      version='0.8',
      description='Fit arbitrary distribution function with emcee',
      author='Christopher Lovell, Stephen Wilkins',
      packages=['fitDF'],
      install_requires=['scipy','emcee'],
     )
