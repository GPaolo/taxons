from setuptools import setup

setup(author='Giuseppe Paolo',
      name='rnd_qd',
      version='0.0.1',
      install_requires=[
            'numpy',
            'torch',
            'pytest',
            'gym',
            'pygame',
            'pathos',
            'Box2D',
            'simplejson',
            'pandas',
            'mujoco-py<2.1,>=2.0'
      ]
)