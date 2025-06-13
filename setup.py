import os
import subprocess
from setuptools import setup, find_packages
from setuptools.command.develop import develop


class PostDevelopCommand(develop):
    def run(self):
        develop.run(self)  # Call the standard install
        # self.install_flash_attn_packages()  # Install flash-attn after everything else
        self.create_directories()

    def install_flash_attn_packages(self):
        # Command to install flash-attn with specific version and index URL
        command = "pip install flash-attn>=2.5.7 --no-build-isolation"
        subprocess.check_call(command.split())

    def create_directories(self):
        project_root = os.path.dirname(os.path.abspath(__file__))
        directory_structure = [
            "cache/models",
            "cache/datasets",
            "cache/checkpoints",
            "cache/joblibs",
            "cache/tasks",
        ]
        for directory in directory_structure:
            path = os.path.join(project_root, directory)
            os.makedirs(path, exist_ok=True)


# Reading requirements from 'requirements.txt'
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="cdpo",
    version="0.1",
    packages=find_packages(),
    cmdclass={
        "develop": PostDevelopCommand,
    },
    py_modules=["cdpo_cli"],
    install_requires=requirements,
    entry_points={
        "console_scripts": ["cdpo=cdpo_cli:cli"]  # Pointing to the cli function in cli.py
    },
    # Other metadata
)
