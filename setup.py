#!/bin/python

import os
from setuptools import setup
import rankcomp

this_dir=os.path.abspath(os.path.diename(__file__))
pj=lambda *path: os.path.abspath(os.path.join(*path))
##read file
read_file=lambda file_name: open(pj(this_dir,file_name),encoding="utf-8").read()
##get relation
read_req=lambda file_name: [line.strip() for line in read_file(file_name).splitlines() if not line.startwith("#")]

setup(
    name="rankcomp",
    python_requires="3.6.5",
    version="0.0.1",
    description="high fided extraction of differential expression genes without considering the batch effects",
    long_description=read_file("README.md"),
    long_description_content_typ="text_markdown",
    author="Terminator Jun",
    author_email="2300869361@qq.com",
    url="https://github.com/SpiderClub/haipproxy",
    install_requires=read_req("needed_packages.txt"),
    include_package_data=True,
    license="MIT",
    keywords=["rank","DEG","batch-effect","rankcomp"],
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ], 

)

