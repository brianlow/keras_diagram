from setuptools import setup
import pypandoc

setup(name='keras_diagram',
      version='1.0.1',
      description='Keras models as ASCII diagrams',
      long_description=pypandoc.convert('README.md','rst',format='markdown'),
      url='http://github.com/brianlow/keras_diagram',
      author='Brian Low',
      author_email='brian.low22@gmail.com',
      license='MIT',
      packages=['keras_diagram'],
      keywords='keras ascii diagram model',
      zip_safe=False)
