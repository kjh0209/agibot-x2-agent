from setuptools import setup

package_name = 'robot_agent'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jensen.kvg',
    maintainer_email='cupid26code@gmail.com',
    description='Agibot X2 conversational agent node',
    license='MIT',
    entry_points={
        'console_scripts': [
            'agent_node = robot_agent.agent_node:main',
            'chat_cli = robot_agent.chat_cli:main',
        ],
    },
)
