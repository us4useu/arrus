import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="arius",
    version="0.0.1",
    author="us4us Ltd.",
    author_email="piotr.jarosik@us4us.eu",
    description="Arius Software Development Kit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/us4us/arius-sdk",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Embedded Systems",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Medical Science Apps."
    ],
    python_requires='>=3.7',
)