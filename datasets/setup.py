from setuptools import setup, find_packages

setup(name='hexucf',
      version='0.0.6-alpha',
      description='UCF Dataset',
      long_description='UCF',
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6'
      ],
      keywords='ucf ai datasets',
      url='',
      author='Vladislav Ovchinnikov',
      author_email='vladovchinnikov950@gmail.com',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'patool',
          'pandas',
          'Pillow',
          'opencv-python',
          'numpy'
      ],
      include_package_data=True,
      zip_safe=False)
