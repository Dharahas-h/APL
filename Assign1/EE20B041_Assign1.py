import sys

def printS(spicelines):
    for x in spicelines :
        for y in x:                                          #printing "logic"
            print(y, end=" ")
        print("")

def tokens(line):
    tokens = line.split("#")[0].split()
    tokLen = len(tokens)
    if (tokLen >= 4):
        elementName = tokens[0]
        node1 = tokens[1]
        node2 = tokens[2]                                    #Analysing and determining tokens
        value = tokens[-1]
        if (tokLen == 4):
            return [value, node2, node1, elementName]
    if (tokLen >= 5) :
        voltageSource = tokens[-2]
        if (tokLen == 5) :
            return [value, voltageSource, node2, node1, elementName]
    if (tokLen == 6):
        node3 = tokens[-3]
        node4 = tokens[-2]
        if (tokLen == 6):
            return [value, node4, node3, node2, node1, elementName]

CIRCUIT = '.circuit'
END = '.end'                     

if (len(sys.argv) != 2):
    print("\nUsage : %s <inputfile>(.netlist)" %sys.argv[0])    #Checking for argument in command line
    exit()

try :
    with open(sys.argv[1], "r") as f :
        lines = f.readlines()
        start = 0
        end = 0
        Spicelines = []
        for line in lines:
            if (CIRCUIT == line.split("#")[0].split("\n")[0]):
                start = lines.index(line)                       #determining starting and ending points 
            if (END == line.split("#")[0].split("\n")[0]):
                end = lines.index(line)
                break

        if (start == end):
            print("Invalid circuit definition")              
            exit(0)

        for line in reversed(lines[start+1:end]):
            linedata = tokens(line)                             #extracting tokens
            Spicelines.append(linedata) 

        printS(Spicelines)
except IOError :
    print("Invalid file")

