from setuptools import setup, find_packages, find_namespace_packages
import logging

p0 = find_packages(where="src")
# print(p0)
# logging.critical(p0)
# p1 = find_namespace_packages(where='vendor.*')
# print(p1)
# logging.critical(p1)
p2 = find_namespace_packages(
    where="src",
    include=["hydra_plugins.*"],
)
# logging.critical(p2)

setup(
    # packages= p0 + p1,
    packages=p0 + p2,
    package_dir={
        "": "src",
        # "pyttitools-adabins": "vendor",
        # "pyttitools-gma": "vendor",
    },
    install_requires=["pyttitools-adabins", "pyttitools-gma"],
)
