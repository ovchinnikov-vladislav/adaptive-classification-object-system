from setuptools import setup, find_packages

setup(name='capsnet_all',
      version='0.0.1-alpha',
      description='capsnets',
      long_description='capsnets',
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6'
      ],
      keywords='capsnet ai classification',
      url='',
      author='Vladislav Ovchinnikov',
      author_email='vladovchinnikov950@gmail.com',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'tensorflow==1.15',
          'keras==2.1.5'
      ],
      include_package_data=True,
      zip_safe=False)