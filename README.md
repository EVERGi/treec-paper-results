
To get the submodules for the boptest case:

git submodule update --init --recursive

To execute the boptest simulator:
make build TESTCASE=bestest_hydronic_heat_pump PORT=5000
make run TESTCASE=bestest_hydronic_heat_pump PORT=5000
