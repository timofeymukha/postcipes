from setuptools import setup
from setuptools import find_packages

setup(name='postcipes',
      version='0.0.1',
      description='Reciepies for post-processing CFD data with turbulucid',
      url='https://github.com/timofeymukha/postcipes',
      author='Timofey Mukha',
      author_email='timofey.mukha@it.uu.se',
      packages=find_packages(),
      entry_points = {
          'console_scripts':[
              'copyFromTimeDirs=postcipes.bin.copyFormTimeDirs:main'
                            ]
      },
      install_requires=[
                    'numpy',
                    'scipy',
                    'matplotlib',
                    'turbulucid'
                       ],
      license="MIT Licence",
      classifiers=[
          "Development Status :: 4 - Beta",
          "License :: OSI Approved :: MIT Licence"
      ],
      zip_safe=False)

