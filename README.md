
## Installation
To be able to run the results, you should first get the git submodules for the BOPTEST simulator with:

```
git submodule update --init --recursive
```

Then install the dependencies by running:

```
pip install -r requirements.txt
```

To execute the boptest simulator, you need to build and run the simulator with:
```
cd boptest/project1-boptest
make build TESTCASE=bestest_hydronic_heat_pump PORT=5000
make run TESTCASE=bestest_hydronic_heat_pump PORT=5000
```

## Example

Use the treec_results.py command line tool to reproduce the results, a description of all the options of the tool is available in the --help menu of the tool
