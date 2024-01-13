from setuptools import setup

_REQUIRED = [
    "tqdm",
    "openai",
    "manifest-ml",
    "pandas",
    "snorkel",
    "cvxpy",
    "bs4",
    "snorkel-metal",
    "tensorboardX",
    "numpy == 1.20.3",
    "networkx == 2.3"
]

setup(
    name="evaporate",
    version="0.0.1",
    description="evaporating data lakes with foundation models",
    author="simran brandon sabri avanika andrew immanuel chris",
    packages=["evaporate"],
    install_requires=_REQUIRED,
)
