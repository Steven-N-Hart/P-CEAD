import subprocess
from distutils.command.build import build as _build  # type: ignore
import setuptools
import logging


# This class handles the pip install mechanism.
class build(_build):  # pylint: disable=invalid-name
    sub_commands = _build.sub_commands + [("CustomCommands", None)]


CUSTOM_COMMANDS = [
    ["echo", "Custom command worked!"],
    ["apt-get", "update"],
    ["apt-get", "--assume-yes", "install", "openslide-tools"],
    ["apt-get", "--assume-yes", "install", "python-openslide"],
    ["pip3", "install", "--upgrade", "pip"],
    ["pip3", "install", "opencv-python-headless"],
    ["pip3", "install", "openslide-python"],
    ["pip3", "install", "matplotlib"],
    ["pip3", "install", "scikit-image"],
    ["pip3", "install", "shapely"]
]


class CustomCommands(setuptools.Command):
    """A setuptools Command class able to run arbitrary commands."""
    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def RunCustomCommand(self, command_list):
        logging.info(f"Running command: {command_list}")
        p = subprocess.Popen(
            command_list,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT)
        # Can use communicate(input="y\n".encode()) if the command run requires
        # some confirmation.
        stdout_data, _ = p.communicate()
        logging.info(stdout_data)
        if p.returncode != 0:
            raise RuntimeError(
                f"Command {command_list} failed: exit code: {p.returncode}")

    def run(self):
        setup_pip()
        setup_apt()
        for command in CUSTOM_COMMANDS:
            self.RunCustomCommand(command)


def setup_apt():
    logging.info("Updating Debian sources.list")
    with open("/etc/apt/sources.list", "w") as f:
        f.write("""
deb [trusted=yes] https://artifactory.mayo.edu/artifactory/deb-virtual buster main
deb [trusted=yes] https://artifactory.mayo.edu/artifactory/deb-docker-remote buster stable
deb [trusted=yes] https://artifactory.mayo.edu/artifactory/deb-virtual buster/updates main
deb [trusted=yes] https://artifactory.mayo.edu/artifactory/deb-virtual buster-updates main
deb [trusted=yes] https://artifactory.mayo.edu/artifactory/deb-virtual gcsfuse-buster main
deb [trusted=yes] https://artifactory.mayo.edu/artifactory/deb-virtual google-compute-engine-buster-stable main
deb [trusted=yes] https://artifactory.mayo.edu/artifactory/deb-virtual google-cloud-packages-archive-keyring-buster main
""")
    logging.info("Debian sources.list has been updated")

def setup_pip():
    logging.info("Updating pip.conf")
    with open("/etc/pip.conf", "w") as pip_conf:
        # Set /etc/pip.conf
        pip_conf.write("""
        [global]
        index-url = https://artifactory.mayo.edu/artifactory/api/pypi/pypi-remote/simple
        """)
    logging.info("pip.conf has been updated")

REQUIRED_PACKAGES = [
]

setuptools.setup(
    name="beam_image_stitch",
    version="0.0.1",
    description="Beam image stitch.",
    install_requires=REQUIRED_PACKAGES,
    packages=setuptools.find_packages(),
    cmdclass={
        # Command class instantiated and run during pip install scenarios.
        "build": build,
        "CustomCommands": CustomCommands,
    }
)
