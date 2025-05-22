import fileinput
import numpy as np

slices = np.arange(300,1900,50)

for k in range(len(slices)-1):
    for i, line in enumerate(fileinput.input(f"performance_jetslice{slices[k]}.cc", inplace=True)):
        if(i==704): 
            print("if(pt>0 && jet_pt>{} && jet_pt<{} && jet_eta<140 && jet_eta>-140 && (jet_eta>-999 && etadiffs>-999)){{".format(slices[k], slices[k+1]), end='')
        else:
            print('{}'.format(line), end='')