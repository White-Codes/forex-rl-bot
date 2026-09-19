from setuptools import setup, find_packages

setup(
    name="forex-rl-bot",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "stable-baselines3==2.3.2",
        "gymnasium==0.29.1",
        "torch==2.2.0",
        "pandas==2.2.0",
        "numpy==1.26.4",
        "yfinance==0.2.36",
        "pyyaml==6.0.1",
        "loguru==0.7.2",
        "matplotlib==3.8.2",
        "tqdm==4.66.1",
    ],
    dependency_links=[
        "git+https://github.com/twopirllc/pandas-ta.git@development#egg=pandas-ta"
    ]
)
