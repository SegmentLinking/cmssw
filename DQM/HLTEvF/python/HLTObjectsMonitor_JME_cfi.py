import FWCore.ParameterSet.Config as cms

jmeObjects = cms.VPSet(
       cms.PSet(
           pathNAME = cms.string("HLT_PFJet60"),
           moduleNAME = cms.string("hltSinglePFJet60"),
           label  = cms.string("PF jet60 (AK4)"),
           xTITLE = cms.string("PF jet (AK4)"),
           etaBINNING    = cms.vdouble(-3.,-2.5,-2.0,-1.5,-1.0,-0.5,0.,0.5,1.0,1.5,2.0,2.5,3.),
           ptBINNING     = cms.vdouble(0.,30.,40.,50.,60.,70.,80.,90.,100.,200.,300.,350.,400.,430.,450.,460.,470.,480.,490.,500.,510.,520.,530.,540.,550.,600.,700.,800.),
           phiBINNING    = cms.vdouble(-3.2,-3.,-2.8,-2.6,-2.4,-2.2,-2.0,-1.8,-1.6,-1.4,-1.2,-1.0,-0.8,-0.6,-0.4,-0.2,0.,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.8,3.0,3.2),
           massBINNING = cms.vdouble(0.,20.,40.,60.,70.,80.,90.,100.,110.,120.,130.,140.,150.,160.,170.,180.,190.,200.,210.),
           dxyBINNING    = cms.vdouble(),
           dzBINNING    = cms.vdouble(),
           dimassBINNING = cms.vdouble(),
           displayInPrimary_eta      = cms.bool(True),
           displayInPrimary_phi      = cms.bool(True),
           displayInPrimary_pt       = cms.bool(True),
           displayInPrimary_mass     = cms.bool(False),
           displayInPrimary_energy   = cms.bool(False),
           displayInPrimary_csv      = cms.bool(False),
           displayInPrimary_etaVSphi = cms.bool(True),
           displayInPrimary_pt_HEP17 = cms.bool(False),
           displayInPrimary_pt_HEM17 = cms.bool(False),
           displayInPrimary_MR       = cms.bool(False),
           displayInPrimary_RSQ      = cms.bool(False),
           displayInPrimary_dxy      = cms.bool(False),
           displayInPrimary_dz       = cms.bool(False),
           displayInPrimary_dimass   = cms.bool(False),                      
           doPlot2D    = cms.untracked.bool(True),
           doPlotETA   = cms.untracked.bool(True),
           doPlotMASS  = cms.untracked.bool(True),
           doPlotENERGY = cms.untracked.bool(False),
           doPlotHEP17 = cms.untracked.bool(True),
           doPlotCSV   = cms.untracked.bool(False),
           doCALO      = cms.untracked.bool(False),
           doPF        = cms.untracked.bool(False),
           doPlotRazor = cms.untracked.bool(False),
           doPlotDXY    = cms.untracked.bool(False),
           doPlotDZ     = cms.untracked.bool(False),
           doPlotDiMass = cms.untracked.bool(False),
       ),
       cms.PSet(
           pathNAME = cms.string("HLT_PFJet200"),
           moduleNAME = cms.string("hltSinglePFJet200"),
           label  = cms.string("PF jet200 (AK4)"),
           xTITLE = cms.string("PF jet (AK4)"),
           etaBINNING    = cms.vdouble(-3.,-2.5,-2.0,-1.5,-1.0,-0.5,0.,0.5,1.0,1.5,2.0,2.5,3.),
           ptBINNING     = cms.vdouble(0.,100.,120.,130.,140.,150.,160.,170.,180.,190.,200.,250.,300.,400.,500.,600.,700.,800.),
           phiBINNING    = cms.vdouble(-3.2,-3.,-2.8,-2.6,-2.4,-2.2,-2.0,-1.8,-1.6,-1.4,-1.2,-1.0,-0.8,-0.6,-0.4,-0.2,0.,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.8,3.0,3.2),
           massBINNING = cms.vdouble(0.,20.,40.,60.,70.,80.,90.,100.,110.,120.,130.,140.,150.,160.,170.,180.,190.,200.,210.),
           dxyBINNING    = cms.vdouble(),
           dzBINNING    = cms.vdouble(),
           dimassBINNING = cms.vdouble(),
           displayInPrimary_eta      = cms.bool(True),
           displayInPrimary_phi      = cms.bool(True),
           displayInPrimary_pt       = cms.bool(True),
           displayInPrimary_mass     = cms.bool(False),
           displayInPrimary_energy   = cms.bool(False),
           displayInPrimary_csv      = cms.bool(False),
           displayInPrimary_etaVSphi = cms.bool(True),
           displayInPrimary_pt_HEP17 = cms.bool(False),
           displayInPrimary_pt_HEM17 = cms.bool(False),
           displayInPrimary_MR       = cms.bool(False),
           displayInPrimary_RSQ      = cms.bool(False),
           displayInPrimary_dxy      = cms.bool(False),
           displayInPrimary_dz       = cms.bool(False),
           displayInPrimary_dimass   = cms.bool(False),                      
           doPlot2D    = cms.untracked.bool(True),
           doPlotETA   = cms.untracked.bool(True),
           doPlotMASS  = cms.untracked.bool(True),
           doPlotENERGY = cms.untracked.bool(False),
           doPlotHEP17 = cms.untracked.bool(True),
           doPlotCSV   = cms.untracked.bool(False),
           doCALO      = cms.untracked.bool(False),
           doPF        = cms.untracked.bool(False),
           doPlotRazor = cms.untracked.bool(False),
           doPlotDXY    = cms.untracked.bool(False),
           doPlotDZ     = cms.untracked.bool(False),
           doPlotDiMass = cms.untracked.bool(False),
       ),
       cms.PSet(
           pathNAME = cms.string("HLT_PFJet450"),
           moduleNAME = cms.string("hltSinglePFJet450"),
           label  = cms.string("PF jet450 (AK4)"),
           xTITLE = cms.string("PF jet (AK4)"),
           etaBINNING    = cms.vdouble(-3.,-2.5,-2.0,-1.5,-1.0,-0.5,0.,0.5,1.0,1.5,2.0,2.5,3.),
           ptBINNING     = cms.vdouble(0.,100.,200.,300.,350.,400.,430.,450.,460.,470.,480.,490.,500.,510.,520.,530.,540.,550.,600.,700.,800.),
           phiBINNING    = cms.vdouble(-3.2,-3.,-2.8,-2.6,-2.4,-2.2,-2.0,-1.8,-1.6,-1.4,-1.2,-1.0,-0.8,-0.6,-0.4,-0.2,0.,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.8,3.0,3.2),
           massBINNING = cms.vdouble(0.,20.,40.,60.,70.,80.,90.,100.,110.,120.,130.,140.,150.,160.,170.,180.,190.,200.,210.),
           dxyBINNING    = cms.vdouble(),
           dzBINNING    = cms.vdouble(),
           dimassBINNING = cms.vdouble(),
           displayInPrimary_eta      = cms.bool(True),
           displayInPrimary_phi      = cms.bool(True),
           displayInPrimary_pt       = cms.bool(True),
           displayInPrimary_mass     = cms.bool(False),
           displayInPrimary_energy   = cms.bool(False),
           displayInPrimary_csv      = cms.bool(False),
           displayInPrimary_etaVSphi = cms.bool(True),
           displayInPrimary_pt_HEP17 = cms.bool(False),
           displayInPrimary_pt_HEM17 = cms.bool(False),
           displayInPrimary_MR       = cms.bool(False),
           displayInPrimary_RSQ      = cms.bool(False),
           displayInPrimary_dxy      = cms.bool(False),
           displayInPrimary_dz       = cms.bool(False),
           displayInPrimary_dimass   = cms.bool(False),                      
           doPlot2D    = cms.untracked.bool(True),
           doPlotETA   = cms.untracked.bool(True),
           doPlotMASS  = cms.untracked.bool(True),
           doPlotENERGY = cms.untracked.bool(False),
           doPlotHEP17 = cms.untracked.bool(True),
           doPlotCSV   = cms.untracked.bool(False),
           doCALO      = cms.untracked.bool(False),
           doPF        = cms.untracked.bool(False),
           doPlotRazor = cms.untracked.bool(False),
           doPlotDXY    = cms.untracked.bool(False),
           doPlotDZ     = cms.untracked.bool(False),
           doPlotDiMass = cms.untracked.bool(False),
       ),
       cms.PSet(
           pathNAME = cms.string("HLT_CaloJet500_NoJetID"),
           moduleNAME = cms.string("hltSinglePFJet450"),
           label  = cms.string("CALO jet500 (AK4)"),
           xTITLE = cms.string("CALO jet (AK4)"),
           etaBINNING    = cms.vdouble(-3.,-2.5,-2.0,-1.5,-1.0,-0.5,0.,0.5,1.0,1.5,2.0,2.5,3.),
           ptBINNING     = cms.vdouble(0.,100.,200.,300.,350.,400.,430.,450.,460.,470.,480.,490.,500.,510.,520.,530.,540.,550.,600.,700.,800.),
           phiBINNING    = cms.vdouble(-3.2,-3.,-2.8,-2.6,-2.4,-2.2,-2.0,-1.8,-1.6,-1.4,-1.2,-1.0,-0.8,-0.6,-0.4,-0.2,0.,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.8,3.0,3.2),
           massBINNING   = cms.vdouble(),
           dxyBINNING    = cms.vdouble(),
           dzBINNING    = cms.vdouble(),
           dimassBINNING = cms.vdouble(),
           displayInPrimary_eta      = cms.bool(True),
           displayInPrimary_phi      = cms.bool(True),
           displayInPrimary_pt       = cms.bool(True),
           displayInPrimary_mass     = cms.bool(False),
           displayInPrimary_energy   = cms.bool(False),
           displayInPrimary_csv      = cms.bool(False),
           displayInPrimary_etaVSphi = cms.bool(True),
           displayInPrimary_pt_HEP17 = cms.bool(False),
           displayInPrimary_pt_HEM17 = cms.bool(False),
           displayInPrimary_MR       = cms.bool(False),
           displayInPrimary_RSQ      = cms.bool(False),
           displayInPrimary_dxy      = cms.bool(False),
           displayInPrimary_dz       = cms.bool(False),
           displayInPrimary_dimass   = cms.bool(False),                      
           doPlot2D    = cms.untracked.bool(True),
           doPlotETA   = cms.untracked.bool(True),
           doPlotMASS  = cms.untracked.bool(False),
           doPlotENERGY = cms.untracked.bool(False),
           doPlotHEP17 = cms.untracked.bool(True),
           doPlotCSV   = cms.untracked.bool(False),
           doCALO      = cms.untracked.bool(False),
           doPF        = cms.untracked.bool(False),
           doPlotRazor = cms.untracked.bool(False),
           doPlotDXY    = cms.untracked.bool(False),
           doPlotDZ     = cms.untracked.bool(False),
           doPlotDiMass = cms.untracked.bool(False),
       ),
       cms.PSet(
           pathNAME = cms.string("HLT_AK8PFJet60"),
           moduleNAME = cms.string("hltSinglePFJet60AK8"),
           label  = cms.string("PF jet60 (AK8)"),
           xTITLE = cms.string("PF jet (AK8)"),
           etaBINNING    = cms.vdouble(-3.,-2.5,-2.0,-1.5,-1.0,-0.5,0.,0.5,1.0,1.5,2.0,2.5,3.),
           ptBINNING     = cms.vdouble(0.,30.,40.,50.,60.,70.,80.,90.,100.,200.,300.,350.,400.,430.,450.,460.,470.,480.,490.,500.,510.,520.,530.,540.,550.,600.,700.,800.),
           phiBINNING    = cms.vdouble(-3.2,-3.,-2.8,-2.6,-2.4,-2.2,-2.0,-1.8,-1.6,-1.4,-1.2,-1.0,-0.8,-0.6,-0.4,-0.2,0.,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.8,3.0,3.2),
           massBINNING = cms.vdouble(0.,20.,40.,60.,70.,80.,90.,100.,110.,120.,130.,140.,150.,160.,170.,180.,190.,200.,210.),
           dxyBINNING    = cms.vdouble(),
           dzBINNING    = cms.vdouble(),
           dimassBINNING = cms.vdouble(),
           displayInPrimary_eta      = cms.bool(True),
           displayInPrimary_phi      = cms.bool(True),
           displayInPrimary_pt       = cms.bool(True),
           displayInPrimary_mass     = cms.bool(False),
           displayInPrimary_energy   = cms.bool(False),
           displayInPrimary_csv      = cms.bool(False),
           displayInPrimary_etaVSphi = cms.bool(True),
           displayInPrimary_pt_HEP17 = cms.bool(False),
           displayInPrimary_pt_HEM17 = cms.bool(False),
           displayInPrimary_MR       = cms.bool(False),
           displayInPrimary_RSQ      = cms.bool(False),
           displayInPrimary_dxy      = cms.bool(False),
           displayInPrimary_dz       = cms.bool(False),
           displayInPrimary_dimass   = cms.bool(False),                      
           doPlot2D    = cms.untracked.bool(True),
           doPlotETA   = cms.untracked.bool(True),
           doPlotMASS  = cms.untracked.bool(True),
           doPlotENERGY = cms.untracked.bool(False),
           doPlotHEP17 = cms.untracked.bool(True),
           doPlotCSV   = cms.untracked.bool(False),
           doCALO      = cms.untracked.bool(False),
           doPF        = cms.untracked.bool(False),
           doPlotRazor = cms.untracked.bool(False),
           doPlotDXY    = cms.untracked.bool(False),
           doPlotDZ     = cms.untracked.bool(False),
           doPlotDiMass = cms.untracked.bool(False),
       ),
       cms.PSet(
           pathNAME = cms.string("HLT_AK8PFJet200"),
           moduleNAME = cms.string("hltSinglePFJet60AK8"),
           label  = cms.string("PF jet200 (AK8)"),
           xTITLE = cms.string("PF jet (AK8)"),
           etaBINNING    = cms.vdouble(-3.,-2.5,-2.0,-1.5,-1.0,-0.5,0.,0.5,1.0,1.5,2.0,2.5,3.),
           ptBINNING     = cms.vdouble(0.,100.,120.,130.,140.,150.,160.,170.,180.,190.,200.,250.,300.,400.,500.,600.,700.,800.),
           phiBINNING    = cms.vdouble(-3.2,-3.,-2.8,-2.6,-2.4,-2.2,-2.0,-1.8,-1.6,-1.4,-1.2,-1.0,-0.8,-0.6,-0.4,-0.2,0.,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.8,3.0,3.2),
           massBINNING = cms.vdouble(0.,20.,40.,60.,70.,80.,90.,100.,110.,120.,130.,140.,150.,160.,170.,180.,190.,200.,210.),
           dxyBINNING    = cms.vdouble(),
           dzBINNING    = cms.vdouble(),
           dimassBINNING = cms.vdouble(),
           displayInPrimary_eta      = cms.bool(True),
           displayInPrimary_phi      = cms.bool(True),
           displayInPrimary_pt       = cms.bool(True),
           displayInPrimary_mass     = cms.bool(False),
           displayInPrimary_energy   = cms.bool(False),
           displayInPrimary_csv      = cms.bool(False),
           displayInPrimary_etaVSphi = cms.bool(True),
           displayInPrimary_pt_HEP17 = cms.bool(False),
           displayInPrimary_pt_HEM17 = cms.bool(False),
           displayInPrimary_MR       = cms.bool(False),
           displayInPrimary_RSQ      = cms.bool(False),
           displayInPrimary_dxy      = cms.bool(False),
           displayInPrimary_dz       = cms.bool(False),
           displayInPrimary_dimass   = cms.bool(False),                      
           doPlot2D    = cms.untracked.bool(True),
           doPlotETA   = cms.untracked.bool(True),
           doPlotMASS  = cms.untracked.bool(True),
           doPlotENERGY = cms.untracked.bool(False),
           doPlotHEP17 = cms.untracked.bool(True),
           doPlotCSV   = cms.untracked.bool(False),
           doCALO      = cms.untracked.bool(False),
           doPF        = cms.untracked.bool(False),
           doPlotRazor = cms.untracked.bool(False),
           doPlotDXY    = cms.untracked.bool(False),
           doPlotDZ     = cms.untracked.bool(False),
           doPlotDiMass = cms.untracked.bool(False),
       ),
       cms.PSet(
           pathNAME = cms.string("HLT_AK8PFJet450"),
           moduleNAME = cms.string("hltSinglePFJet450AK8"),
           label  = cms.string("PF jet500 (AK8)"),
           xTITLE = cms.string("PF jet (AK8)"),
           etaBINNING    = cms.vdouble(-3.,-2.5,-2.0,-1.5,-1.0,-0.5,0.,0.5,1.0,1.5,2.0,2.5,3.),
           ptBINNING     = cms.vdouble(0.,100.,200.,300.,350.,400.,430.,450.,460.,470.,480.,490.,500.,510.,520.,530.,540.,550.,600.,700.,800.),
           phiBINNING    = cms.vdouble(-3.2,-3.,-2.8,-2.6,-2.4,-2.2,-2.0,-1.8,-1.6,-1.4,-1.2,-1.0,-0.8,-0.6,-0.4,-0.2,0.,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.8,3.0,3.2),
           massBINNING = cms.vdouble(0.,20.,40.,60.,70.,80.,90.,100.,110.,120.,130.,140.,150.,160.,170.,180.,190.,200.,210.),
           dxyBINNING    = cms.vdouble(),
           dzBINNING    = cms.vdouble(),
           dimassBINNING = cms.vdouble(),
           displayInPrimary_eta      = cms.bool(True),
           displayInPrimary_phi      = cms.bool(True),
           displayInPrimary_pt       = cms.bool(True),
           displayInPrimary_mass     = cms.bool(True),
           displayInPrimary_energy   = cms.bool(False),
           displayInPrimary_csv      = cms.bool(False),
           displayInPrimary_etaVSphi = cms.bool(True),
           displayInPrimary_pt_HEP17 = cms.bool(False),
           displayInPrimary_pt_HEM17 = cms.bool(False),
           displayInPrimary_MR       = cms.bool(False),
           displayInPrimary_RSQ      = cms.bool(False),
           displayInPrimary_dxy      = cms.bool(False),
           displayInPrimary_dz       = cms.bool(False),
           displayInPrimary_dimass   = cms.bool(False),                      
           doPlot2D    = cms.untracked.bool(True),
           doPlotETA   = cms.untracked.bool(True),
           doPlotMASS  = cms.untracked.bool(True),
           doPlotENERGY = cms.untracked.bool(False),
           doPlotHEP17 = cms.untracked.bool(True),
           doPlotCSV   = cms.untracked.bool(False),
           doCALO      = cms.untracked.bool(False),
           doPF        = cms.untracked.bool(False),
           doPlotRazor = cms.untracked.bool(False),
           doPlotDXY    = cms.untracked.bool(False),
           doPlotDZ     = cms.untracked.bool(False),
           doPlotDiMass = cms.untracked.bool(False),
       ),
)

