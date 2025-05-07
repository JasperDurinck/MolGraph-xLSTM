from setuptools import setup, find_packages
import os

here = os.path.abspath(os.path.dirname(__file__))

# with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
#     long_description = "\n" + fh.read()

VERSION = '0.0.1.0'
DESCRIPTION = ""
LONG_DESCRIPTION = ''


# Setting up
setup(
    name="molgraph_xlstm",
    version=VERSION,
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description='', 
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'molgraph_xlstm': ['version.json'] 
    },
    entry_points={
        'console_scripts': [
        ]
    },
    install_requires=[], 
    keywords=[],
    classifiers=[],
    
    include_dirs=[]

)