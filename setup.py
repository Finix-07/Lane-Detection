"""
Lane Detection Package Setup
"""
from setuptools import setup, find_packages

setup(
    name="lane-detection",
    version="1.0.0",
    description="Lane Detection with Bezier Curves and SegFormer",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.30.0",
        "pillow>=9.0.0",
        "scipy>=1.10.0",
        "matplotlib>=3.5.0",
        "tqdm>=4.65.0",
        "numpy>=1.23.0",
    ],
    extras_require={
        "dev": [
            "tensorboard>=2.13.0",
            "opencv-python>=4.7.0",
            "jupyter>=1.0.0",
            "pytest>=7.0.0",
        ]
    },
    python_requires=">=3.8",
)
