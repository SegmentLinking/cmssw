#!/usr/bin/env python3
# Configuration test of customizeHLTforMkFitAlpaka: the procModifier applies the device chain in every HLT_75e33 menu
# that defines it, the SerialSync clones come with alpakaValidationHLT, and configurations outside the supported
# envelope are refused with a RuntimeError instead of being ignored. The modifiers of a python process cannot change
# and the menu modules are shared by every load, so every scenario runs in its own interpreter.
import subprocess
import sys

MENUS = ('HLT_75e33_cff', 'HLT_75e33_timing_cff', 'HLT_75e33_trackingOnly_cff')
SCENARIOS = ['procModifier:' + menu for menu in MENUS] + ['customise', 'validation', 'alpakaAlone', 'otherFitConfig',
                                                          'stripQualityDB', 'noBuilding', 'v2p2Building',
                                                          'cloneEngineBuilding', 'standardBuilding']


def runScenario(scenario):
    import FWCore.ParameterSet.Config as cms
    from Configuration.Eras.Era_Phase2C22I13M9_cff import Phase2C22I13M9
    from Configuration.ProcessModifiers.alpakaValidationHLT_cff import alpakaValidationHLT
    from Configuration.ProcessModifiers.trackingMkFitAlpaka_cff import trackingMkFitAlpaka
    from Configuration.ProcessModifiers.trackingMkFitFit_cff import trackingMkFitFit
    import RecoTracker.MkFitAlpaka.customizeHLTforMkFitAlpaka as mkFitAlpaka

    # label -> module type after the customise; the menu modules it replaces must be gone
    deviceChain = {mkFitAlpaka.OT_DEVICE: 'MkFitAlpakaOTRecHitsProducer@alpaka',
                   mkFitAlpaka.EOH: 'MkFitAlpakaEventOfHitsProducer@alpaka',
                   mkFitAlpaka.BUILD_DEVICE: 'MkFitAlpakaBuildProducer@alpaka',
                   mkFitAlpaka.FIT_DEVICE: 'MkFitAlpakaFitDeviceProducer@alpaka',
                   mkFitAlpaka.OT_HITS: 'MkFitAlpakaPhase2ClusterIndexToHit',
                   'hltInitialStepTracks': 'MkFitAlpakaOutputTrackConverter'}
    replaced = (mkFitAlpaka.BUILD, mkFitAlpaka.FIT, mkFitAlpaka.PIX_HITS)
    failures = []

    def check(condition, what):
        if not condition:
            failures.append(what)
        print('%s %s' % ('ok  ' if condition else 'FAIL', what))

    def types(process, labels):
        return {label: getattr(process, label).type_() if hasattr(process, label) else None for label in labels}

    def menuProcess(menu, *modifiers):
        process = cms.Process('HLTX', Phase2C22I13M9, *modifiers)
        process.load('HLTrigger.Configuration.' + menu)
        return process

    def checkDeviceChain(process, what):
        check(types(process, deviceChain) == deviceChain, '%s: device chain %s' % (what, types(process, deviceChain)))
        left = [label for label in replaced if hasattr(process, label)]
        check(not left, '%s: menu modules removed (left: %s)' % (what, left))

    def refused(build, reason, what):
        try:
            build()
        except RuntimeError as error:
            check(reason in str(error), '%s: refused (%s)' % (what, error))
            return
        check(False, '%s: accepted' % what)

    def customisedWith(modify):
        process = menuProcess(MENUS[0], trackingMkFitFit)
        modify(process)
        return mkFitAlpaka.customizeHLTforMkFitAlpaka(process)

    if scenario.startswith('procModifier:'):
        menu = scenario.split(':')[1]
        checkDeviceChain(menuProcess(menu, trackingMkFitFit, trackingMkFitAlpaka), menu + ' with the procModifier')
    elif scenario == 'customise':
        # --customise on a menu loaded with trackingMkFitFit only; a second application changes nothing
        process = mkFitAlpaka.customizeHLTforMkFitAlpaka(menuProcess(MENUS[0], trackingMkFitFit))
        checkDeviceChain(process, MENUS[0] + ' with --customise')
        dump = process.dumpPython()
        checkDeviceChain(mkFitAlpaka.customizeHLTforMkFitAlpaka(process), MENUS[0] + ' customised twice')
        check(process.dumpPython() == dump, MENUS[0] + ' customised twice: configuration unchanged')
    elif scenario == 'validation':
        process = menuProcess(MENUS[0], trackingMkFitFit, trackingMkFitAlpaka, alpakaValidationHLT)
        checkDeviceChain(process, MENUS[0] + ' with alpakaValidationHLT')
        # SerialSync clones of the device chain (not of the OT index map): alpaka_serial_sync:: device modules
        serial = {label + mkFitAlpaka.SERIAL: ('alpaka_serial_sync::' + moduleType.removesuffix('@alpaka')
                                               if moduleType.endswith('@alpaka') else moduleType)
                  for label, moduleType in deviceChain.items() if label != mkFitAlpaka.OT_HITS}
        check(types(process, serial) == serial, MENUS[0] + ' with alpakaValidationHLT: SerialSync clones %s'
              % types(process, serial))
    elif scenario == 'alpakaAlone':
        refused(lambda: menuProcess(MENUS[0], trackingMkFitAlpaka), 'trackingMkFitFit', 'trackingMkFitAlpaka alone')
    elif scenario == 'otherFitConfig':
        refused(lambda: customisedWith(lambda process: setattr(getattr(process, mkFitAlpaka.FIT), 'config',
                                                               cms.ESInputTag('', 'anotherIterationConfig'))),
                'config', 'menu fit with another iteration config')
    elif scenario == 'stripQualityDB':
        refused(lambda: customisedWith(lambda process: setattr(getattr(process, mkFitAlpaka.EOH),
                                                               'useStripStripQualityDB', cms.bool(True))),
                'useStripStripQualityDB', 'strip quality DB on')
    elif scenario == 'noBuilding':
        refused(lambda: customisedWith(lambda process: delattr(process, mkFitAlpaka.BUILD)), mkFitAlpaka.BUILD,
                'process without the mkFit building')
    elif scenario == 'v2p2Building':
        # MkFinderV2p2 on the device: the device building module gets the menu's routine, v2p2 in the menu
        process = customisedWith(lambda process: None)
        checkDeviceChain(process, 'menu building with its own buildingRoutine')
        routine = getattr(process, mkFitAlpaka.BUILD_DEVICE).buildingRoutine.value()
        check(routine == 'v2p2', 'menu building with its own buildingRoutine: device routine %s' % routine)
    elif scenario == 'cloneEngineBuilding':
        # the clone engine stays selectable: the menu building with buildingRoutine cloneEngine
        process = customisedWith(lambda process: setattr(getattr(process, mkFitAlpaka.BUILD), 'buildingRoutine',
                                                         cms.string('cloneEngine')))
        checkDeviceChain(process, 'menu building with buildingRoutine cloneEngine')
        routine = getattr(process, mkFitAlpaka.BUILD_DEVICE).buildingRoutine.value()
        check(routine == 'cloneEngine', 'menu building with buildingRoutine cloneEngine: device routine %s' % routine)
    elif scenario == 'standardBuilding':
        # the device chain ports the clone engine and MkFinderV2p2 only
        refused(lambda: customisedWith(lambda process: setattr(getattr(process, mkFitAlpaka.BUILD), 'buildingRoutine',
                                                               cms.string('standard'))),
                'buildingRoutine', 'menu building with buildingRoutine standard')
    else:
        check(False, 'unknown scenario ' + scenario)
    return len(failures)


if len(sys.argv) > 1:
    sys.exit(1 if runScenario(sys.argv[1]) else 0)
failed = [scenario for scenario in SCENARIOS if subprocess.call([sys.executable, __file__, scenario]) != 0]
print('failed scenarios: %s' % failed if failed else 'all scenarios passed')
sys.exit(1 if failed else 0)
