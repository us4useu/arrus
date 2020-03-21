import setuptools

setuptools.setup(
    name="arius",
    version="0.2.0rc1",
    author="us4us Ltd.",
    author_email="piotr.jarosik@us4us.eu",
    description="Arius Software Development Kit",
    long_description="Arius Software Development Kit",
    long_description_content_type="text/markdown",
    url="https://us4us.eu",
    packages=setuptools.find_packages(),
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
        "PyYAML==5.1.2",
        "scipy>=1.3.1"
    ],
    package_data={
        'arius': ['devices/*.pyd', 'devices/*.lib']
    },
    python_requires='>=3.7',
)
