from setuptools import setup, find_packages

setup(
    name="tcml-workflow",
    version="1.0.0",
    description="Thermal Conductivity ML workflow: structure-sensitive κ_L prediction from composition and TDA descriptors, trained on Phonix",
    author="Yen-Ju Wu",
    author_email="wu.yenju@nims.go.jp",
    url="https://github.com/yjwu-sci/tcml-workflow",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.0",
        "scipy>=1.10",
        "scikit-learn>=1.3",
        "xgboost>=2.0",
        "pymatgen>=2024.1",
        "ase>=3.22",
        "spglib>=2.1",
        "pyarrow>=14",
        "joblib>=1.3",
        "tqdm>=4.65",
    ],
    extras_require={
        "tda": ["gudhi>=3.9", "persim>=0.3"],
        "dev": ["jupyter", "pytest>=7", "ruff>=0.1"],
        "mp": ["mp-api>=0.40"],
    },
)
