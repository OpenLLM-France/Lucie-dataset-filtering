# -*- coding: utf-8 -*-
from setuptools import setup

package_dir = {"": "src"}

packages = ["blmrdata"]

package_data = {"": ["*"]}

install_requires = [
    "tqdm",
    "jsonlines",
    "loguru",
    "click",
    "sentencepiece>=0.1.82",
    "kenlm @ git+https://github.com/kpu/kenlm.git@master",
    "redpajama @ git+https://github.com/EvanDufraisse/RedPajamaV2-Utils@main",
]

dependency_links = []

entry_points = {"console_scripts": ["blmrdata = blmrdata.cli:cli"]}

setup_kwargs = {
    "name": "blmrdata",
    "version": "0.1.0",
    "description": "",
    "long_description": f"Lucie-dataset-filtering: Code to compile and preprocess training data for Lucie's training",
    "author": "OpenLLM",
    "author_email": "contact@openllm-france.fr",
    "maintainer": "None",
    "maintainer_email": "None",
    "url": "None",
    "package_dir": package_dir,
    "packages": packages,
    "package_data": package_data,
    "install_requires": install_requires,
    "entry_points": entry_points,
    "dependency_links": dependency_links,
    "python_requires": ">=3.8",
}


setup(**setup_kwargs)
