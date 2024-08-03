"""Setup."""

from setuptools import setup

setup(
    name='ramannoodle',
    version='1.0.0-alpha',
    description='Raman spectra from first principles calculations.',
    long_description='Raman spectra from first principles calculations. Supports VASP.',
    author="Willis O'Leary",
    author_email='wolearyc@gmail.com',
    license='MIT',
    packages=['ramannoodle'],
    zip_safe=False,
    install_requires=[
        'numpy',
        'scipy',
        'spglib',
    ],
)
