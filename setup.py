import bpepp
from pathlib import Path

from setuptools import setup

long_description = Path('README.md').read_text(encoding='utf-8', errors='ignore')

classifiers = [  # copied from https://pypi.org/classifiers/
    'Development Status :: 5 - Production/Stable',
    'Intended Audience :: Developers',
    'Topic :: Utilities',
    'Topic :: Text Processing',
    'Topic :: Text Processing :: General',
    'Topic :: Text Processing :: Filters',
    'Topic :: Text Processing :: Linguistic',
    'License :: University of Southern California (USC) Restricted License',
    'Programming Language :: Python :: 3',
]

setup(
    name='bpepp',
    version=bpepp.__version__,
    description=bpepp.__description__,
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='University of Southern California (USC) Restricted License',
    classifiers=classifiers,
    python_requires='>=3.7',
    url='https://github.com/thammegowda/bpepp',
    download_url='https://github.com/thammegowda/bpepp',
    platforms=['any'],
    author='Thamme Gowda',
    author_email='tgowdan@gmail.com',
    py_modules=['bpepp'],  # for a package dir with __init__.py use packages=['bpepp'],
    entry_points={
        'console_scripts': ['bpepp=bpepp:main'],
    },
    install_requires=['tqdm']
)