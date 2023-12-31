from setuptools import setup

with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(
    name='unsupervised',
    version='0.3.0',
    description='Functions for applying dimensionality reduction',
    author='johan vargas',
    author_email='example@email.com',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=[
        'unsupervised',
        'unsupervised.python',
        'unsupervised.python.dimensionality_reduction',
    ],
    install_requires=['numpy'],
)
