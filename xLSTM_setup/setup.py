from setuptools import setup, find_packages
from pathlib import Path
from torch.utils import cpp_extension

#adjusted setup for non conflicting dependencies (or non needed dependencies from the orignal setup)

setup(
    name="xlstm",
    version="2.0.2",
    description="A novel LSTM variant with promising performance compared to Transformers.",
    long_description='',
    long_description_content_type="text/markdown",
    url="https://github.com/NX-AI/xlstm",
    project_urls={
        "Source Code": "https://github.com/NX-AI/xlstm",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(exclude=["test*", "res*", "experiments*"]),
    python_requires=">=3.9",
    include_package_data=True,
    package_data={
        "": [
            "blocks/slstm/src/cuda/*.c",
            "blocks/slstm/src/cuda/*.cc",
            "blocks/slstm/src/cuda/*.h",
            "blocks/slstm/src/cuda/*.cu",
            "blocks/slstm/src/cuda/*.cuh",
            "blocks/slstm/src/util/*.c",
            "blocks/slstm/src/util/*.cc",
            "blocks/slstm/src/util/*.h",
            "blocks/slstm/src/util/*.cu",
            "blocks/slstm/src/util/*.cuh",
        ]
    },
)

