from setuptools import setup

url = ""
version = "0.1.0"
readme = open('README.md').read()

setup(
    name="feedWebGL2",
    packages=["feedWebGL2"],
    version=version,
    description="Tools for implementing WebGL2 feedback processing stages for graphics preprocessing or other purposes",
    long_description=readme,
    include_package_data=True,
    author="Aaron Watters",
    author_email="awatters@flatironinstitute.org",
    url=url,
    install_requires=[
        "jp_doodle",
        "jp_proxy_widget",
        ],
    license="MIT"
)
