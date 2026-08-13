from setuptools import setup

setup(name='dynamicpumps',
      version='1.0.0',
      description='Python package for sizing and analysis of dynamic pumps',
      url='https://github.com/janstruzinski/DynamicPumps.git',
      author='Jan Struzinski',
      license='GPL-3.0-only',
      py_modules=['BarskePump', 'PumpSystem', 'Fluid', 'functions'],
      package_dir={'': 'DynamicPumps'},
      install_requires=['CoolProp>=7.2.0', 'matplotlib>=3.10.8', 'numpy>=2.4.3', 'scipy>=1.17.1',
                        'tabulate>=0.10.0'],
      python_requires='>=3.14',
      keywords='dynamic pump centrifugal Barske partial emission inducer turbopump',
      zip_safe=False,
      )
