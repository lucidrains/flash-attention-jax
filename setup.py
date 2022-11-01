from setuptools import setup, find_packages

setup(
  name = 'flash-attention-jax',
  packages = find_packages(exclude=[]),
  version = '0.2.0',
  license='MIT',
  description = 'Flash Attention - in Jax',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/lucidrains/flash-attention-jax',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'transformers',
    'attention mechanism',
    'jax'
  ],
  install_requires=[
    'einops',
    'jax>=0.2.20'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
