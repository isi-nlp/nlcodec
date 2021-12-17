# Importing nlcodec requires other packages to be installed already . Prevents installation .
# Adding version and description here as a hack.
# import nlcodec
from pathlib import Path

from setuptools import setup, find_packages

long_description = Path('README.md').read_text(encoding='utf-8', errors='ignore')

version = '0.4.0'
description = """nlcodec is a collection of encoding schemes for natural language sequences. 
nlcodec.db is a efficient storage and retrieval layer for integer sequences of varying lengths."""

classifiers = [  # copied from https://pypi.org/classifiers/
    'Development Status :: 5 - Production/Stable',
    'Intended Audience :: Developers',
    'Topic :: Utilities',
    'Topic :: Text Processing',
    'Topic :: Text Processing :: General',
    'Topic :: Text Processing :: Filters',
    'Topic :: Text Processing :: Linguistic',
    'License :: OSI Approved :: Apache Software License',
    'Programming Language :: Python :: 3',
]

setup(
    name='nlcodec',
    version=version,
    description=description,
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='Apache Software License 2.0',
    classifiers=classifiers,
    python_requires='>=3.7',
    url='https://github.com/pegasus-lynx/nlcodec/tree/mwe_schemes',
    download_url='https://github.com/pegassus-lynx/nlcodec/tree/mwe_schemes',
    platforms=['any'],
    author='Thamme Gowda',
    author_email='tgowdan@gmail.com',

    packages=find_packages(exclude=['experimental']),
    entry_points={
        'console_scripts': [
            'nlcodec=nlcodec.__main__:main',
            'nlcodec-learn=nlcodec.learn:main',
            'nlcodec-db=nlcodec.bitextdb:main',
            'nlcodec-freq=nlcodec.term_freq:main'
        ],
    },
    install_requires=['tqdm', 'numpy']
)
