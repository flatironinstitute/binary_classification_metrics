from setuptools import setup

url = ""
version = "0.1.0"
readme = open('README.md').read()

setup(
    name="binary_classification_metrics",
    packages=["binary_classification_metrics"],
    version=version,
    description="Visualizations and other code for exploring binary classification metrics.",
    long_description=readme,
    include_package_data=True,
    author="Aaron Watters",
    author_email="awatters@flatironinstitute.org",
    url=url,
    install_requires=[
        "jp_doodle",
        "jp_proxy_widget",
        "feedWebGL2",
        "h5py",
        ],
    license="MIT"
)
