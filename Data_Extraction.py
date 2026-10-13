
filename = "BGO_Cs_0_10cm.Spe"
with open(filename, "r") as data_file:
    flight_data = data_file.readlines().strip('\n')
print(flight_data[12:])   