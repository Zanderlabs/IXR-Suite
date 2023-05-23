# IXR Suite

Middleware between physiological hardware and primarily VR applications.**This package is build and tested on python 3.10.x**

## Dependencies

### Windows

Dependencies can be installed using `requirements.txt`

``` shell
pip install -r requirements.txt
```

### Linux

For other platforms then Windows the `liblsl` binary (part of the `pylsl`) is not included in the python package. Recommended approach is to use `conda` (environments) at this point. More information can be found [here](https://docs.conda.io/projects/conda/en/latest/index.html).

1. Create an `conda` environment with python 3.10
2. Activate your enviroment.
3. Install the `liblsl` binary in he `conda` environment.
4. Proceed with installing the regular dependencies.

```shell
conda create --name=<name_of_choice> python=3.10
conda activate <name_of_choise>
conda install -c conda-forge liblsl
pip install -r requirements.txt
```
