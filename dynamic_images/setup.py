from setuptools import setup, find_packages

setup(name='hex_dynamic_images',
      version='0.0.1-alpha',
      description='UCF Dataset',
      long_description='UCF',
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6'
      ],
      keywords='dynamic images ai datasets',
      url='',
      author='Vladislav Ovchinnikov',
      author_email='vladovchinnikov950@gmail.com',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'opencv-python',
          'numpy'
      ],
      include_package_data=True,
      zip_safe=False)
