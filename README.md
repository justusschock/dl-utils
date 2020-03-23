# Python Template Repository

This repository contains a fully-functionable package structure including (empty) tests.

It's features include (but are not limited to):
* An already working package structure
* A working requirement handling
* Minimal effort pypi releases
* Pre-Configured CI/CD (With Travis)
* Code coverage analysis
* Python Code Style Checks

> If you want to add something to this repo, please submit a PR. Contributions are very welcome.

## Customize it!

To customize this repo, you need to have a look at the following chapters.

### Directory-Name
You might want to customize your package-name.

To do this, you simply have to rename the `template-repo` directory to whatever you want.
 > Make sure, to also change it in [line 37 of your setup.py](setup.py#L37), or you won't be able to install your package anymore!

### Python Package Metadata

To customize your python package, you just have to change your `setup.py`.

Currently the important part looks like 
```python
setup(
    name='template_package',
    version=_version,
    packages=find_packages(),
    url='https://github.com/justusschock/template-repo-python',
    test_suite="unittest",
    long_description=readme,
    long_description_content_type='text/markdown',
    install_requires=requirements,
    tests_require=["coverage"],
    python_requires=">=3.5",
    author="Justus Schock",
    author_email="justus.schock@rwth-aachen.de",
    license=license,
)
```
This includes the default information for me and must be adjusted to your needs:

* `name` provides the package-name you can later import
* `version` provides the package-version (which will currently be extracted from your package's `__init__.py`, but be also set manually)
* `packages` is a list defining all packages (and their sub-packages and the sub-packages of their sub-packages and so on...), that should be installed. This is automatically extracted by `find_packages`, which also accepts some sub-packages to ignore (besides some other arguments).
`url` specifies the packages homepage (in this case the current GitHub repo); You might want to change it to your repos homepage.
* `test_suite` defines the test-suite to use for your unittests. In this repo template, we'll python's built-in framework `unittest`, but you can change this too; *Just make sure to also change this, when we get to CI/CD.*
* `long_description` does what it sayes: It provides a long description of your package. Currently this is parsed from your `README.md`
* `long_description_content_type` defines your description type; I set it to markdown in most cases
* `install_requires` is a list containing all your package requirements. They are automatically parsed from a `requirements.txt` file
* `tests_require` does the same thing for your unittests.
* `python_requires` specifies the python version, your package can be installed to (here it's been set to python 3.5 or above, since this is what I usually use). *Depending on the version you specify here, you might not be able to use all of python's latest features*
* `author` and `author_email` specify who you are.
* `license` specifies the license you want to release your code with. This is parsed from a `LICENSE` file.

There are still many other options to include here, but these are the most basic ones.

### Unittests
If you want to add/change some unit-tests, you should do this in a new python file starting with `test_`. [Here](https://docs.python.org/3/library/unittest.html) is a good introduction on how to write unittests with the `unittest` framework. After you added these tests, you may run them with either `coverage run -m unittest`or `python -m unittest`.

They are basically doing the same, but `coverage` additionally checks, how many of your code-lines are currently covered by your tests.

The unittests are also automatically triggered within [CI/CD](#cicd)

### Specifying Codecov
The [`.codecov.yml`](.codecov.yml) file specifies, how coverage should behave, how to calculate the coverage (i.e. what files to include for line counting) etc. 

### Requirements
If you want to add new requirements, simply add them to the [`requirements.txt`](requirements/install.txt) file.

### Packaging on PyPi
If you plan to release your package on pypi, ship wheels for it, you might need the [`MANSIFEST.in`](MANIFEST.in) file, since it specifies (among other things), which files to include to your binaries.

### Setup.cfg
The [`setup.cfg`](setup.cfg) file currently only specifies, which directories to exclude from style checking.

### Gitignore
The `.gitignore` file is a real life saver. It prevents files and directories that match certain patterns from being added to your git repository, when you push new stuff to it. You may append more specific patterns here.

### CI/CD
Now, we talked a lot about CI/CD. This repository uses [`travis`](https://travis-ci.com) as CI/CD and per default simply runs tests and style checks for your code.

To use this feature, you have to enable travis for your repository.

#### YAMl-Specifications
The [`.travis.yml`](.travis.yml) file specifies the CI/CD behavior. Currently it only runs tests and style-checks  with Python 3.7 on Linux Xenial. You may also include additional cases to the test matrix or add deployment (e.g. deploying your docs to GitHub Pages or similar stuff).

#### Scripts
The scripts used b CI/CD to install the requirements and run your tests are lying at [`scripts/ci`](scripts/ci).
The file names indicate pretty well, what tey're doing. Of course you can customize them too.

If you want Travis to automatically fix your code style where possible you have to add a github access token to travis, comment in the [lines 6-28](scripts/ci/run_style_checks.sh#L6-L28) and change the environment variable and the repository in [line 27](scripts/ci/run_style_checks.sh#L27).

