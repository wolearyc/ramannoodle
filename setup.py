"""Setup."""

from setuptools import setup
from pathlib import Path


# read the contents of README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()


setup(
    name='ramannoodle',
    version='v0.1.1-alpha',
    description='Helps calculate Raman spectra from first-principles calculations.',
    long_description=long_description,
    long_description_content_type='text/markdown',
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
