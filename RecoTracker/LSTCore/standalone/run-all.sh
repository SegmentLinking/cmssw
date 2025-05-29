#lstFiles=(temp_aux/lst_ttbar.cc temp_aux/lst_ttbar_cut.cc temp_aux/lst_QCD_flat.cc temp_aux/lst_QCD.cc)
lstFiles=(temp_aux/lst_QCD.cc) # I CHANGED IT TO THE UNCUT VERSION

#perfFiles=(temp_aux/performance_eff.cc temp_aux/performance_eff_zoom.cc temp_aux/performance_jetslice50.cc temp_aux/performance_jetslice100.cc temp_aux/performance_jetslice150.cc temp_aux/performance_jetslice200.cc temp_aux/performance_jetslice250.cc temp_aux/performance_jetslice300.cc temp_aux/performance_Rslice1.cc temp_aux/performance_Rslice4.cc)
perfFiles=(temp_aux/performance_jetslice1750.cc)

#sampleNames=(ttbar ttbar-pt750 FlatPT QCD1800-2400PU0)
sampleNames=(QCD1800-2400PU0)

for i in ${!sampleNames[@]}; do
    cp ${lstFiles[$i]} bin/lst.cc

    for j in ${!perfFiles[@]}; do
        cp ${perfFiles[$j]} efficiency/src/performance.cc

        rm -r LST*.root
        lst_make_tracklooper -mcC;
        lst -i ${sampleNames[$i]} -p 0.6 -l -o LSTNtuple.root;
        createPerfNumDenHists -i LSTNtuple.root -o LSTNumDen.root;
        python3 efficiency/python/lst_plot_performance.py --individual --pt_cut 0.9 LSTNumDen.root -t "${sampleNames[$i]}29--slicing"; #"${sampleNames[$i]}$j--slicing";

        echo "${sampleNames[$i]} and ${perfFiles[$j]} done." > log.txt
    done
done