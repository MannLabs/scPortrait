import setuptools
import platform
import os
import sys

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# get path for requirements file
parent_folder = os.path.dirname(os.path.realpath(__file__))
requirementPath = os.path.join(parent_folder, "requirements.txt")

# load requirements
install_requires = []
if os.path.isfile(requirementPath):
    with open(requirementPath) as f:
        install_requires = f.read().splitlines()

setuptools.setup(
    name="sparcscore",
    version="1.0.0",
    author="Georg Wallmann, Sophia Mädler, Niklas Schmacke",
    author_email="maedler@biochem.mpg.de",
    description="SPARCSpy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MannLabs/SPARCSspatial",
    project_urls={
        "Bug Tracker": "https://github.com/MannLabs/SPARCSspatial",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    include_package_data=True,
    zip_safe=True,
    install_requires=install_requires,
)

if platform.system() == "Linux":
    target_folder = "/usr/local/bin"
    commands = ["sparcs-stat", "sparcs-split", "sparcs-merge"]
    src_directory = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "src", "sparcscmd"
    )
    bin_directory = os.path.dirname(os.path.abspath(sys.executable))

    for cmd in commands:
        src_module = os.path.join(src_directory, cmd + ".py")
        symlink_origin = os.path.join(bin_directory, cmd)

        # make script executebale
        st = os.stat(src_module)
        os.chmod(src_module, st.st_mode | 0o111)

        if not os.path.islink(symlink_origin):
            print(f"symlink for {cmd} does not exist, will be created")
            os.symlink(src_module, symlink_origin)


else:
    print(
        "Automatic symlinks are only supported on linux. Please add sparcs cli commands to your PATH."
    )
