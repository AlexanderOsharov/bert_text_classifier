from setuptools import setup, find_packages

setup(
    name='bert_text_classifier',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'transformers',
        'scikit-learn',
        'torch',
        'scikit-learn',
        'seaborn',
        'matplotlib',
    ],
    package_data={
        'bert_text_classifier': ['data/dataset.json']
    }
)