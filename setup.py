from setuptools import setup, find_packages

setup(
    name="PINN-kit",
    version="0.0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "torch",
        "matplotlib",
        "scikit-optimize",
    ],
    python_requires=">=3.9",
)
