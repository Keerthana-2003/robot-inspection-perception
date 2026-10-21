from setuptools import setup

package = "inspection_perception_ros"
setup(
    name=package,
    version="0.1.0",
    packages=[package],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package}"]),
        (f"share/{package}", ["package.xml"]),
        (f"share/{package}/launch", ["launch/inspection.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Keerthana Mandadi",
    maintainer_email="keerthanam2k3@gmail.com",
    description="ROS 2 wrapper for the robot inspection perception pipeline",
    license="MIT",
    entry_points={"console_scripts": [f"inspection_node = {package}.node:main"]},
)
