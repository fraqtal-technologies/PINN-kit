from setuptools import setup, find_packages

setup(
    name="PINN-kit",
    version="0.0.2",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.21.0",
        "torch>=2.0.0",
        "matplotlib>=3.5.0",
        "scikit-optimize>=0.9.0",
    ],
    python_requires=">=3.9",
)
