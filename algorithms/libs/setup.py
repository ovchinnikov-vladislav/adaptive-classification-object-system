from setuptools import setup, find_packages

setup(name='libs-ovchinnikov-capsnets',
      version='0.0.3-alpha',
      description='В данной библиотеке находятся вспомогательные классы и функции для реализации и использования'
                  'различных архитектур капсульных сетей',
      long_description='В данной библиотеке находятся вспомогательные классы и функции для реализации и использования'
                       'различных архитектур капсульных сетей',
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6'
      ],
      keywords='capsnets ai classification',
      url='',
      author='Vladislav Ovchinnikov',
      author_email='vladovchinnikov950@gmail.com',
      license='Apache 2.0',
      packages=find_packages(),
      install_requires=[
          'opencv-python',
          'patool',
          'pandas',
          'Pillow',
          'tensorflow==2.3.0'
      ],
      include_package_data=True,
      zip_safe=False)
