# IXR Suite

The IXR Suite is a middleware between physiological hardware and virtual environments. It is the result of a collaboration between [Zanderlabs](https://zanderlabs.com/) and the [»Young Investigator Group – Intuitive XR«](https://www.b-tu.de/researchschool/gefoerderte-forschungsaktivitaeten/young-investigator-group-intuitive-xr) of the Brandenburg University of Technology Cottbus-Senftenberg.

## How-to guide

If you just want to run the IXR Suite and not develop anything, you can download the latest version on the [release page](https://github.com/Zanderlabs/IXR-Suite/releases).

See [this slidedeck](https://docs.google.com/presentation/d/1aDD0TuLVqdQ3OrO-flbkxTzb2gyvidsCp46c-4gSyvE) for a simple guide on how to install and run the software on Windows with VS Code if you want to contribute to development.

## Applications
- [Real Virtual Magic Skyrim VR mod](https://www.nexusmods.com/skyrimspecialedition/mods/58489)
- [Unity API](https://github.com/Zanderlabs/IXR4Unity)

## Dependencies

**This package is build and tested on python 3.12.x**

### Windows

Dependencies can be installed using `requirements.txt`

``` shell
pip install -r requirements.txt
```

### Linux

For other platforms then Windows the `liblsl` binary (part of the `pylsl`) is not included in the python package. Recommended approach is to use `conda` (environments) at this point. More information can be found [here](https://docs.conda.io/projects/conda/en/latest/index.html).

1. Create an `conda` environment with python 3.12
2. Activate your enviroment.
3. Install the `liblsl` binary in he `conda` environment.
4. Proceed with installing the regular dependencies.

```shell
conda create --name=<name_of_choice> python=3.12
conda activate <name_of_choise>
conda install -c conda-forge liblsl
pip install -r requirements.txt
```
