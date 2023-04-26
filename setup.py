import os
from setuptools import setup, find_packages


def read_reqs(name):
    with open(os.path.join(os.path.dirname(__file__), name)) as f:
        return [line for line in f.read().split("\n") if line and not line.strip().startswith("#")]


setup(
    name="tvault",
    version="0.3.16",
    description="Log and find diffs between pytorch models",
    author="Saeyoon Oh",
    author_email="david@vessl.ai",
    url="https://github.com/vessl-ai/tvault.git",
    license="MIT",
    packages=find_packages("src"),
    package_dir={"": "src"},
    include_package_data=True,
    entry_points={"console_scripts": ["tvault = tvault.__init__:cli_main"]},
    install_requires=read_reqs("requirements.txt"),
    python_requires=">=3",
)
