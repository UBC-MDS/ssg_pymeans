from distutils.core import setup

setup(
    name='ssg_pymeans',
    version='3.0',
    author='Sophia Wang, Susan Fung, Guanchen Zhang',
    packages=['ssg_pymeans'],
    url='https://github.com/UBC-MDS/ssg_pymeans',
    description='A Python package for k-means clustering',
    long_description=open('README.txt').read(),
    install_requires=['numpy', 'pandas', 'matplotlib'],
    include_package_data=True,
    package_data={
        'ssg_pymeans': ['data/*.csv'],
    }
)
