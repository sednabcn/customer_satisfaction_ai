import versioneer
from setuptools import find_packages, setup

setup(
    name="simleng_ai",  # unknown option
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="A module to evaluate Data_Analysis for customer_satisfaction starategies",
    long_description="Data Analysis-Machine Learning",
    long_description_content_type="text/x-rst",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        # License, see https://choosealicense.com
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    url=[
        "https://github.com/sednabcn/customer_satisfaction_ai",
        "https://github.com/sednabcn/customer_satisfaction_ai/issues",
    ],
    # entry_points={
    #    "console_scripts": [""],
    # },
    author="Ruperto Pedro Bonet Chaple",
    author_email="ruperto.bonet@modelphysmat.com",
    License=["MIT"],
    # packages=["customer_satisfaction_ai"],
    readme="Readme.md",  # unknown option
    python_requires=">=3.8",
    keywords="customer_satisfaction data_analisys setuptools development",
    # manifest_in=["./input_file/customer_satisfaction.txt","./datasets/*"],
    install_requires=[
        "biokit >=0.5.0",
        "colormap>=1.0.4",
        "matplotlib>=3.7.1",
        "numpy>=1.24.4",  # 1.24.4
        "packaging>=23",  # 21.3
        "pandas>=2.0.3",  # 2.0.3
        "scipy==1.10.1",  # 1.8.0
        "statsmodels>=0.14.0",  # 0.14.0
        "scikit-learn>=1.3.0",
    ],
    test_suite="nose.collector",
    tests_require=["nose"],
    scripts=["  "],
    include_package_data=True,
    zip_safe=False,
)
