import sys
import numpy as np
import cmath
import math

CIRCUIT = '.circuit'
END = '.end'
dc = 1


class RCL:
    def __init__(self, tokens):
        self.name = tokens[0]                                             #Defining classes for resistor, capacitor and inductor 
    
        self.node1 = (tokens[1])
        self.node2 = (tokens[2])
        if (tokens[0][0] == "R"):
            self.value = tokens[3]
        elif (tokens[0][0] == "C"):
            if (frequency != 0):
                self.value = complex(0, -1/(frequency*(tokens[3])))
        elif (tokens[0][0] == "L"):
            if (frequency != 0):
                self.value = complex(0, frequency*(tokens[3]))        

class VI:
    def __init__(self, tokens):
        self.name = tokens[0]
        if (tokens[1] == "GND"):
            tokens[1] = "0"
        if (tokens[2] == "GND"):                                           #Defining classes for voltage and current sources 
            tokens[2] = "0"
        self.node1 = int(tokens[1])
        self.node2 = int(tokens[2])
        self.type = tokens[3]
        if (tokens[3] == "ac"):
            self.value = cmath.rect(float(tokens[4])/2, float(tokens[5]))
        elif (tokens[3] == "dc"):
            self.value = float(tokens[4])
        else:
            self.value = float(tokens[3])

Resistors = []
Capacitors = []
Inductors = []
VSource = []
CSource = []

with open(sys.argv[1], "r") as f:
    lines = f.readlines()
    
    start = 1
    end = 1
    frequency = 0
    for line in lines :
        string = line.split("#")[0].split("\n")[0]
        if (string == CIRCUIT):
            start = lines.index(line)                                 #Reading the netlist file
        if (string == END):
            end = lines.index(line)
        if (string[:3] == ".ac"):
            frequency = 2*(math.pi)*float(string.split()[-1])
            if (frequency != 0):
                dc=0

    if (start==end):
        print("Invalid Circuit Definition")
        exit(0)

    for line in lines[start+1 : end]:
        tokens = line.split("#")[0].split("\n")[0].split()
        if (tokens[1] == 'GND'):
            tokens[1] = "0"
        if (tokens[2] == 'GND'):
            tokens[2] = "0"
        tokens[1] = int(tokens[1])
        tokens[2] = int(tokens[2])                            
        if (tokens[0][0] == "R"):
            tokens[3] = float(tokens[3])                                #creating component objects
            R = RCL(tokens)
            Resistors.append(R)
        elif (tokens[0][0] == "C"):
            tokens[3] = float(tokens[3])
            C = RCL(tokens)
            Capacitors.append(C)
        elif (tokens[0][0] == "L"):
            tokens[3] = float(tokens[3])
            L = RCL(tokens)
            Inductors.append(L)
        elif (tokens[0][0] == "V"):
            V = VI(tokens)
            VSource.append(V)
        elif (tokens[0][0] == "I"):
            I = VI(tokens)
            CSource.append(I)

k=len(VSource)                                                  #k = no. of voltage sources = no. of current variables


equal = {}                                                      #n = no. of nodes  
n=0
if (dc == 1):                                                   #In case of DC the circiut is solved for steady state values
    equal = {}
    for x in Inductors:
        if (x.node1 > x.node2):
            equal[x.node1] = x.node2
        else:                                                   #Inductors in case of DC(frequency = 0) equate the node voltages it is connected across
            equal[x.node2] = x.node1                            #Capacitors in case of DC(frequency = 0) are replaced
    for x in Resistors:
        if (x.node1 in equal.keys()):
            x.node1 = equal[x.node1]
        if (x.node2 in equal.keys()):
            x.node2 = equal[x.node2]
        
        if (x.node1 > n):
            n = x.node1
        if(x.node2 > n):
            n = x.node2

    M = np.zeros((n+k+1, n+k+1))                                
    b = np.zeros(n+k+1)
            
    for x in Resistors:
        M[x.node1 , x.node1 ] += 1/x.value
        M[x.node1 , x.node2 ] += -1/x.value
        M[x.node2 , x.node1 ] += -1/x.value
        M[x.node2 , x.node2 ] += 1/x.value
else:
    for y in [Resistors, Capacitors, Inductors, VSource, CSource]:
        for x in y:
            if (x.node1 > n):
                n = x.node1                                             #In case of AC, ciruit is solved with complex numbers
            if (x.node2 > n):
                n = x.node2

    M = np.zeros((n+k+1, n+k+1), dtype = complex)
    b = np.zeros(n+k+1, dtype = complex)
    
    for y in [Resistors, Inductors, Capacitors]:
        for x in y:
            M[x.node1 , x.node1 ] += 1/x.value
            M[x.node1 , x.node2 ] += -1/x.value
            M[x.node2 , x.node1 ] += -1/x.value
            M[x.node2 , x.node2 ] += 1/x.value
        
i=1    
VSource.sort(key = lambda x : int(x.name[1]))

for x in VSource:
    M[x.node1, n+i] += 1
    M[x.node2, n+i] += -1
    M[n+i, x.node1] += 1
    M[n+i, x.node2] += -1
    b[n+i] += x.value
    i = i+1

for x in CSource:
    b[x.node1] += x.value
    b[x.node2] += -x.value

try :
    x= np.linalg.solve(M[1:,1:], b[1:])
except :
    print("It has Infinite solutions")
    exit()

alias = {"V0" : 0}    
p=1 
for i in x:
    if (p < n +1):
        alias["V"+str(p)] = i
        p=p+1
    else:
        alias["I"+str(p - n)] = -i
        p=p+1
for x in equal.keys():
        alias["V"+str(x)] = "V"+str(equal[x])   

print(alias, end="\n")





        
        
            
    
            
            
                
                
