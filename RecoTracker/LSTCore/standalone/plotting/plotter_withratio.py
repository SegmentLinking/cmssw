import argparse
import sys
import os
from array import *
import json
from ROOT import TColor

def efficiencyPalette():
    pcol = []
    for iBin in range(0, 100):
        rgb = []
        if iBin < 70:
            rgb = [0.70 + 0.007 * iBin, 0.00 + 0.0069 * iBin, 0.00]
        elif iBin < 90:
            rgb = [0.70 + 0.007 * iBin, 0.00 + 0.0069 * iBin + 0.10 + 0.01 * (iBin - 70), 0.00]
        else:
            rgb = [0.98 - 0.098 * (iBin - 90), 0.80, 0.00]
        pcol.append(TColor.GetColor(rgb[0], rgb[1], rgb[2]))
    return pcol

parser = argparse.ArgumentParser(description="Takes an input JSON config and extracts plots from ROOT files.")
parser.add_argument("inputJsonConfig", help="Path to the input JSON config file")
parser.add_argument("-f", "--fast", default=0, action="count")
parser.add_argument("-v", "--verbosity", default=1)
args = parser.parse_args()

with open(args.inputJsonConfig, 'r') as f:
    config = json.loads(f.read())

from ROOT import gROOT, TH1, gStyle, TFile, TCanvas, TPad, TLegend, TF1, TLatex, TArrow
gROOT.SetBatch(True)
gROOT.ProcessLine("gErrorIgnoreLevel = 1001")

TH1.AddDirectory(False)
gStyle.SetOptTitle(0)
gStyle.SetPaintTextFormat("1.3f")
gStyle.SetHistMinimumZero()
gStyle.SetOptStat(0)

for keyPlot in config:
    if args.verbosity == 1:
        print('Processing plot config: {}'.format(keyPlot))
        print('Config comment: {}'.format(config[keyPlot]['comment']))

    inputFilenames = []
    inputPlotNames = []
    inputFolders = []
    inputLegendEntries = []
    inputLabels = []

    for keyInputs in config[keyPlot]['inputs']:
        inputFilenames.append(config[keyPlot]['inputs'][keyInputs]['filename'])
        inputPlotNames.append(config[keyPlot]['inputs'][keyInputs]['plot'])
        inputFolders.append(config[keyPlot]['inputs'][keyInputs]['folder'])
        inputLegendEntries.append(config[keyPlot]['inputs'][keyInputs]['legendEntry'])
        inputLabels.append(config[keyPlot]['inputs'][keyInputs]['label'])

    inputHistos = {}
    for iHisto in range(len(inputFilenames)):
        inputFile = TFile.Open(inputFilenames[iHisto])
        if not inputFile:
            print('[ERROR] File not found: {}'.format(inputFilenames[iHisto]))
            sys.exit()
        inputDir = inputFile.GetDirectory(inputFolders[iHisto])
        inputName = None
        for keys in inputDir.GetListOfKeys():
            if keys.GetName() == inputPlotNames[iHisto] or inputPlotNames[iHisto] == "all":
                inputName = keys.GetName()
                if inputName.find("=") > 0 or inputDir.Get(inputName).ClassName() == "TDirectoryFile":
                    continue
                if inputPlotNames[iHisto] == "all":
                    nameTag = inputName
                else:
                    nameTag = "histo"
                if not nameTag in inputHistos:
                    inputHistos[nameTag] = []
                print('Load plot \'{}\': {}'.format(inputFolders[iHisto], inputName))
                print(inputDir.Get(inputName).ClassName())
                histo = inputDir.Get(inputName).Clone(str(iHisto))
                inputHistos[nameTag].append(histo)

    colorMap = [TColor.GetColor(c) if isinstance(c, str) else c for c in config[keyPlot]['plot']['colorMap']]
    markerMap = config[keyPlot]['plot']['markerMap']
    legendRange = config[keyPlot]['plot']['legendRange']
    plotX = config[keyPlot]['plot']['x']
    plotY = config[keyPlot]['plot']['y']
    yTitleOffset = config[keyPlot]['plot']['yTitleOffset']
    if args.verbosity == 1:
        print('Using colormap: {}'.format(colorMap))
        print('Using markermap: {}'.format(markerMap))
    if len(colorMap) < len(inputFolders):
        print('[ERROR] The defined colormap has not enough entries.')
        sys.exit()
    if len(markerMap) < len(inputFolders):
        print('[ERROR] The defined markermap has not enough entries.')
        sys.exit()

    for histoName, histograms in inputHistos.items():
        for iHisto in range(len(histograms)):
            histograms[iHisto].SetLineColor(colorMap[iHisto])
            histograms[iHisto].SetLineWidth(2)
            histograms[iHisto].SetMarkerStyle(markerMap[iHisto])
            histograms[iHisto].SetMarkerColor(colorMap[iHisto])
            histograms[iHisto].SetMarkerSize(1.2)
            histograms[iHisto].GetXaxis().SetLabelSize(0)
            histograms[iHisto].GetYaxis().SetLabelSize(0.042)
            histograms[iHisto].GetYaxis().SetTitleSize(0.051)
            histograms[iHisto].GetYaxis().SetTitleOffset(yTitleOffset)
            histograms[iHisto].GetXaxis().SetTickLength(0)
            histograms[iHisto].GetYaxis().SetTitle(plotY[2])
            histograms[iHisto].GetXaxis().SetRangeUser(plotX[0], plotX[1])
            histograms[iHisto].GetYaxis().SetRangeUser(plotY[0], plotY[1])
        canvas = TCanvas('canvas', 'canvas', 1000, 1000)
        pad1 = TPad('pad1', 'Main pad', 0.01, 0.30, 1.00, 1.00)
        pad2 = TPad('pad2', 'Ratio pad', 0.01, 0.00, 1.00, 0.30)
        pad1.SetBottomMargin(0.02)
        pad2.SetTopMargin(0.05)
        pad2.SetBottomMargin(0.35)
        pad1.Draw()
        pad2.Draw()

        pad1.cd()
        pad1.SetGrid()

        option = config[keyPlot]['plot']['option']

        hasLogX = option.find("logX") > -1
        hasLogY = option.find("logY") > -1
        hasLogRatioY = option.find("logRatioY") > -1
        if hasLogX:
            option = option.replace("logX", "")
            pad1.SetLogx()
        if hasLogY:
            option = option.replace("logY", "")
            pad1.SetLogy()
        if hasLogRatioY:
            option = option.replace("logRatioY", "")
            pad2.SetLogy()

        if hasLogX:
            histograms[0].GetXaxis().SetNdivisions(10, False)

        for iHisto in range(len(histograms)):
            if iHisto == 0:
                histograms[iHisto].Draw(option)
            else:
                histograms[iHisto].Draw('same' + option)

        pad1.Update()
        if len(histograms) > 1:
            pad2.cd()
            pad2.SetGrid()
            if hasLogX:
                pad2.SetLogx()
            h_ref = histograms[0]
            ratioYMin = config[keyPlot]['plot']['yRatio'][0]
            ratioYMax = config[keyPlot]['plot']['yRatio'][1]
            ratios = []
            for iHisto in range(1, len(histograms)):
                ratio = histograms[iHisto].Clone("ratio{}".format(iHisto))
                ratio.Divide(h_ref)
                ratio.SetLineColor(colorMap[iHisto])
                ratio.SetMarkerColor(colorMap[iHisto])
                ratio.SetMarkerStyle(markerMap[iHisto])
                ratio.GetYaxis().SetTitle("Ratio")
                ratio.GetYaxis().SetTitleSize(0.12)
                ratio.GetYaxis().SetTitleOffset(0.44)
                ratio.GetYaxis().SetLabelSize(0.10)
                ratio.GetYaxis().SetRangeUser(ratioYMin, ratioYMax)
                ratio.GetYaxis().SetNdivisions(4, False)
                ratio.GetXaxis().SetRangeUser(plotX[0], plotX[1])
                if hasLogX:
                    ratio.GetXaxis().SetNdivisions(10, False)
                ratio.GetXaxis().SetTitle(plotX[2])
                ratio.GetXaxis().SetTitleSize(0.12)
                ratio.GetXaxis().SetLabelSize(0.10)
                ratio.GetXaxis().SetTitleOffset(1.05)
                ratios.append(ratio)
            for iHisto in range(len(ratios)):
                if iHisto == 0:
                    ratios[iHisto].Draw(option)
                else:
                    ratios[iHisto].Draw('same' + option)
            arrowSize = 0.01
            arrowLength = ratioYMin if hasLogRatioY else (ratioYMax - ratioYMin) * 0.05
            arrows = []
            for iRatio, ratio in enumerate(ratios):
                numerator = histograms[iRatio + 1]
                for iBin in range(1, ratio.GetNbinsX() + 1):
                    val = ratio.GetBinContent(iBin)
                    divByZero = h_ref.GetBinContent(iBin) == 0 and numerator.GetBinContent(iBin) != 0
                    if not divByZero and h_ref.GetBinContent(iBin) == 0 and numerator.GetBinContent(iBin) == 0:
                        continue
                    x = ratio.GetBinCenter(iBin)
                    if x < plotX[0] or x > plotX[1]:
                        continue
                    if divByZero or val > ratioYMax:
                        arr = TArrow(x, ratioYMax - arrowLength, x, ratioYMax, arrowSize, "|>")
                        arr.SetLineColor(ratio.GetLineColor())
                        arr.SetFillColor(ratio.GetLineColor())
                        arr.Draw()
                        arrows.append(arr)
                    elif val < ratioYMin:
                        arr = TArrow(x, ratioYMin + arrowLength, x, ratioYMin, arrowSize, "|>")
                        arr.SetLineColor(ratio.GetLineColor())
                        arr.SetFillColor(ratio.GetLineColor())
                        arr.Draw()
                        arrows.append(arr)
            pad2.Update()
            canvas.cd()

        if len(histograms) > 1:
            leg = TLegend(legendRange[0], legendRange[1], legendRange[2], legendRange[3])
            leg.SetBorderSize(0)
            leg.SetTextFont(43)
            leg.SetTextSize(28)
            leg.SetFillStyle(0)
            leg.SetFillColor(0)
            leg.SetEntrySeparation(0.05)
            leg.SetMargin(0.5)
            leg.SetHeader(config[keyPlot]['plot']['legendTitle'])
            leg.GetListOfPrimitives().First().SetTextFont(63)
            for iHisto in range(len(histograms)):
                leg.AddEntry(histograms[iHisto], inputLegendEntries[iHisto], 'LP')
            pad1.cd()
            leg.Draw()

        canvas.cd()
        latex = TLatex()
        latex.SetNDC()
        latex.SetTextFont(61)
        latex.SetTextSize(0.035)
        latex.SetTextAlign(11)
        latex.DrawLatex(0.115, 0.94, config[keyPlot]['plot']['logo'][0])
        latex.SetTextFont(52)
        latex.SetTextSize(0.030)
        latex.SetTextAlign(11)
        latex.DrawLatex(0.195, 0.94, config[keyPlot]['plot']['logo'][1])
        latex.SetTextFont(42)
        latex.SetTextSize(0.030)
        latex.SetTextAlign(31)
        latex.DrawLatex(0.91, 0.94, config[keyPlot]['plot']['caption'])
        canvas.Update()

        outputDirectory = config[keyPlot]['output']['directory']
        if args.verbosity == 1:
            print('Output directory: {}'.format(outputDirectory))
        if not os.path.exists(outputDirectory):
            os.makedirs(outputDirectory)
        for fileType in config[keyPlot]['output']['fileType']:
            if config[keyPlot]['output']['filenamePlot'] == "all":
                fileNamePlot = histoName
            else:
                fileNamePlot = config[keyPlot]['output']['filenamePlot']
            canvas.SaveAs(os.path.join(outputDirectory, fileNamePlot + '.' + fileType))

    if args.verbosity == 1:
        print('')
