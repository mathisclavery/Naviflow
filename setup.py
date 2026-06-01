"""Installation de Naviflow.

    pip install -e .

Le mode editable (-e) installe le package par lien : tes modifications dans
naviflow/ sont prises en compte immediatement, sans reinstaller.
"""

from setuptools import find_packages, setup

with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="naviflow",
    version="0.2.0",
    description="Prediction d'affluence dans les transports IDF Mobilites",
    packages=find_packages(),  # detecte naviflow et ses sous-packages
    install_requires=requirements,
    python_requires=">=3.10",
)
