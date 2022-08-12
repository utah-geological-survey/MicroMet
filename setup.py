import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Micromet",
    version="0.1.0",
    author="Paul Inkenbrant",
    author_email="paulinkenbrandt@utah.gov",
    description="Scripts to process raw Eddy Covariance data for estimation of evapotranspiration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/utah-geological-survey/MicroMet",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD 3-Clause License",
        "Operating System :: OS Independent",
    ),
)