from setuptools import setup

setup(name='rnd_qd',
      version='0.0.1',
      install_requires=['numpy', 'torch', 'pytest', 'nevergrad'],
      dependency_link=['git+https://github.com/facebookresearch/nevergrad@master#egg=nevergrad -0']
)