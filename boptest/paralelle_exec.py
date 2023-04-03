from boptest.train import train_valid_function
import sys

def setup_boptest_images(port):
    case = int(port)%6

    if case == 0:
        time_period = "peak_heat_day"
        electricity_price = "highly_dynamic"
    elif case == 1:
        time_period = "typical_heat_day"
        electricity_price = "highly_dynamic"
    elif case == 2:
        time_period = "peak_heat_day"
        electricity_price = "dynamic"
    elif case == 3:
        time_period = "typical_heat_day"
        electricity_price = "dynamic"
    elif case == 4:
        time_period = "peak_heat_day"
        electricity_price = "constant"
    elif case == 5:
        time_period = "typical_heat_day"
        electricity_price = "constant"

    train_valid_function(port, electricity_price, time_period, 150)

if __name__ =="__main__":
    setup_boptest_images(sys.argv[1])