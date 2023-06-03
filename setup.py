from setuptools import setup

setup(name='ramannoodle',
      version='1.0',
      description="Calculated Raman spectra from MD",
      long_description="",
      author="Willis O'Leary",
      author_email='wolearyc@mit.edu',
      license='TODO',
      packages=['ramannoodle'],
      zip_safe=False,
      install_requires=[
          'numpy',
          'scipy', 
          'matplotlib',
          'ase',
          'phonopy',
          'spglib',
          'tabulate'
          ],
      )
