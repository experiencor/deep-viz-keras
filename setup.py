from setuptools import setup


def readme():
    with open('README.md') as f:
        return f.read()


setup(name='saliency',
      version='0.1',
      description='Implement Saliency Maps using Keras',
      long_description=readme(),
      classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3',
      ],
      keywords="Saliency maps keras deep learning back propagation",
      url='http://github.com/mzmmoazam/deep-viz-keras',
      author='mzm',
      author_email='mzm.moazam@gmail.com',
      long_description_content_type='text/markdown',
      packages=['saliency'],
      install_requires=[
          'numpy',
          'keras',
          'matplotlib',
          'Pillow',
          'tensorflow'
      ],
      include_package_data=True,
      zip_safe=False)