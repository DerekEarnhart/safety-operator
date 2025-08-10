from setuptools import setup, find_packages


setup(
    name='safety_operator',
    version='0.1.0',
    description='A drop-in safety filter for AI model outputs.',
    author='Derek Earnhart',
    license='MIT',
    packages=find_packages(),
    install_requires=['numpy'],
)
