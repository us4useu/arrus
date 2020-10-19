import setuptools

setuptools.setup(
    name="arrus",
    version="0.5.0",
    author="us4us Ltd.",
    author_email="support@us4us.eu",
    description="API for Research/Remote Ultrasound",
    long_description="ARRUS - API for Research/Remote Ultrasound",
    long_description_content_type="text/markdown",
    url="https://us4us.eu",
    packages=setuptools.find_packages(
        exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    classifiers=[
        "Development Status :: 1 - Planning",

        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",

        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Embedded Systems",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Medical Science Apps."
    ],
    install_requires=[
        "numpy>=1.17.4",
        "scipy>=1.3.1"
    ],
    python_requires='>=3.7',
)
