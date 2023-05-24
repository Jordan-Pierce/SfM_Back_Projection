## Installing Dependencies

This workflow uses python dependencies that do not come installed in Metashape's python environment. This short tutorial will explain the process for installing the dependencies needed to run the wokflow in Metashape's python environment.

The normal process for installing python dependencies to Metashape's python environment can be found [here](https://agisoft.freshdesk.com/support/solutions/articles/31000136860-how-to-install-external-python-module-to-photoscan-professional-pacakge). Metashape's python environment is typically located at the following path:

`"%programfiles%\Agisoft\Metashape Pro\python\python.exe"`

Where `programfiles` represents the `Program Files` folder located in the `C:` drive.

To install the necessary dependencies (and any others you desire), it's recommend to use the `setup.py` script, which will install all of the dependencies in the `requirements.txt` file using `pip`. To do this, simply open a command terminal and run the `setup.py` script with python:

```
python setup.py
```

This will kick off a subprocess and install each of the dependencies.



```python

```
