from vizier.service import pyvizier as vz
print("Supported Algirthms:")
for algo in vz.Algorithm:
    print(algo)