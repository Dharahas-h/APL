import sys

with open(sys.argv[1], "r") as f:
    lines = f.readlines()
    for line in lines :
        print(line)

    for line in lines :
            string = line.split("#")[0].split("\n")[0]
            if (string == ".circuit"):
                start = lines.index(line)
            if (string == ".end"):
                end = lines.index(line)
            if (string[:3] == ".ac"):
                frequency = float(string.split()[-1])
                if (frequency != 0):
                    dc=0
                    print(dc)
                    print(frequency)
   
