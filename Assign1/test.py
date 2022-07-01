import sys

with open(sys.argv[1],"r") as f:
    lines = f.readlines()
    for line in lines:
        if(line.split("#")[0].split()[0][0] == "R"):
            print(line)
    
