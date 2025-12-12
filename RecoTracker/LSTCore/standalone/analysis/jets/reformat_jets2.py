##########################################################
#
# Adds deltaR branches to the trackingNtuple.root file
# using GenJets.
#
##########################################################

import matplotlib.pyplot as plt
import ROOT
from ROOT import TFile
from myjets import getLists, createJets, matchArr
import numpy as np

# Load existing tree
# file =  TFile("/data2/segmentlinking/CMSSW_12_2_0_pre2/trackingNtuple_ttbar_PU200.root")
file = TFile("trackingNtuple_100_GenJet.root")
old_tree = file["trackingNtuple"]["tree"]

# Create a new ROOT file to store the new TTree
new_file = ROOT.TFile("new_tree_100_GenJet2.root", "RECREATE")

# Create a new subdirectory in the new file
new_directory = new_file.mkdir("trackingNtuple")

# Change the current directory to the new subdirectory
new_directory.cd()

# Create a new TTree with the same structure as the old one but empty
new_tree = old_tree.CloneTree(0)  

# Account for bug in 12_2_X branch
new_tree.SetBranchStatus("ph2_bbxi", False) 

# Create a variable to hold the new leaves' data (a list of floats)
new_leaf_deltaEta = ROOT.std.vector('float')()
new_leaf_deltaPhi = ROOT.std.vector('float')()
new_leaf_deltaR = ROOT.std.vector('float')()

# Create a new branch in the tree
new_tree.Branch("sim_deltaEta", new_leaf_deltaEta)
new_tree.Branch("sim_deltaPhi", new_leaf_deltaPhi)
new_tree.Branch("sim_deltaR", new_leaf_deltaR)

# Loop over entries in the old tree
for ind in range(old_tree.GetEntries()):
    old_tree.GetEntry(ind)

    # Clear the vector to start fresh for this entry
    new_leaf_deltaEta.clear()
    new_leaf_deltaPhi.clear()
    new_leaf_deltaR.clear()

    # Creates the lists that will fill the leaves
    simPt = old_tree.sim_pt
    simEta = old_tree.sim_eta
    simPhi = old_tree.sim_phi
    simLen = len(simPt)

    print(old_tree.Print())

    genJetsPt = old_tree.genJetPt
    genJetsEta = old_tree.genJetEta
    genJetsPhi = old_tree.genJetPhi
    genJetLen = len(genJetsPt)

    # Declare arrays (all entries will be filled with non-dummy values)
    deltaEtas = np.ones(simLen)*-999
    deltaPhis = np.ones(simLen)*-999
    deltaRs = np.ones(simLen)*-999
    
    for i in range(len(simPt)):
        dRTemp = 999
        dPhiTemp = 999
        dEtaTemp = 999
        for j in range(genJetLen):
            dEtaj = simEta[i] - genJetsEta[j]
            dPhij = np.arccos(np.cos(simPhi[i] - genJetsPhi[j]))
            dRj = np.sqrt(dEtaj**2 + dPhij**2)
            
            # Selects smallest dR, corresponding to closest jet
            if(dRj < dRTemp): 
                dRTemp = dRj
                dPhiTemp = dPhij
                dEtaTemp = dEtaj
        deltaRs[i] = dRTemp
        deltaPhis[i] = dPhiTemp
        deltaEtas[i] = dEtaTemp
    

    # Add the list elements to the vector
    for value in deltaEtas:
        new_leaf_deltaEta.push_back(value)
    for value in deltaPhis:
        new_leaf_deltaPhi.push_back(value)
    for value in deltaRs:
        new_leaf_deltaR.push_back(value)

    # Fill the tree with the new values
    new_tree.Fill()

# Write the tree back to the file
new_tree.Write()
new_file.Close()
file.Close()

