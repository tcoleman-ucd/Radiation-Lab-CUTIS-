"""Load source spectra for BGO, NaI and CdTe and calibrate each detector."""

from pathlib import Path

from spectra import Spectrum
from detector import Detector
from Efficiency import absolute_efficiency, plot_absolute_efficiency

from Offaxis import run_off_axis , plot_comparison 



Detectors = {
    "BGO": {
        "background": "BGO_Background_180s.Spe",
        "sources": {
            "Am": "BGO_Am_0_10cm_180s.Spe",
            "Ba": "BGO_Ba_0_10cm_300s.Spe",
            "Cs": "BGO_Cs_0_10cm_180s.Spe",
            "Co": "BGO_Co_0_10cm_420s.Spe"
        }
    },

    "NaI": {
        "background": "Nal_Background_420s.Spe",
        "sources": {
            "Am": "NaI_Am_0_10cm_180s.Spe",
            "Ba": "NaI_Ba_0_10cm_300s.Spe",
            "Cs": "NaI_Cs_0_10cm_180s.Spe",
            "Co": "NaI_Co_0_10cm_420s.Spe"
        }
    },

    "CdTe": {
        "background": "CdTe_Background_300s.mca",
        "sources": {
            "Am": "CdTe_Am_0_10cm_180s.mca",
            "Ba": "CdTe_Ba_0_10cm_300s.mca",
            "Cs": "CdTe_Cs_10cm_180s.mca",
            "Co": "CdTe_Co_0_10cm_420s.mca"
        }
    }
}

measurements = {
    "BGO": {
        "background": "BGO_Background_180s.Spe",
        "sources": {"Am":{
    -90: "BGO_Am_90(2)_10cm_180s.Spe",
    -45: "BGO_Am_45(2)_10cm_180s.Spe",
    0:  "BGO_Am_0_10cm_180s.Spe",
    45: "BGO_Am_45(1)_10cm_180s.Spe",
    90: "BGO_Am_90(1)_10cm_180s.Spe",},
    "Ba":{
    -90: "BGO_Ba_90(2)_10cm_300s.Spe",
    -45: "BGO_Ba_45(2)_10cm_300s.Spe",
    0:  "BGO_Ba_0_10cm_300s.Spe",
    45: "BGO_Ba_45(1)_10cm_300s.Spe",
    90: "BGO_Ba_90(1)_10cm_300s.Spe",},
    "Cs":{
    -90: "BGO_Cs_90(2)_10cm_180s.Spe",
    -45: "BGO_Cs_45(2)_10cm_180s.Spe",
    0:  "BGO_Cs_0_10cm_180s.Spe",
    45: "BGO_Cs_45(1)_10cm_180s.Spe",
    90: "BGO_Cs_90(1)_10cm_180s.Spe",},
    "Co":{
    -90: "BGO_Co_90(2)_10cm_420s.Spe",
    -45: "BGO_Co_45(2)_10cm_420s.Spe",
    0:  "BGO_Co_0_10cm_420s.Spe",
    45: "BGO_Co_45(1)_10cm_420s.Spe",
    90: "BGO_Co_90(1)_10cm_420s.Spe",},
    }},
    "NaI": {
            "background": "Nal_Background_420s.Spe",
            "sources": {"Am":{
        -90: "NaI_Am_90(2)_10cm_180s.Spe",
        -45: "NaI_Am_45(1)_10cm_180s.Spe",
        0:  "NaI_Am_0_10cm_180s.Spe",
        45: "NaI_Am_45(1)_10cm_180s.Spe",
        90: "NaI_Am_90(1)_10cm_180s.Spe",},
        "Ba":{
        -90: "NaI_Ba_90(2)_10cm_300s.Spe",
        -45: "NaI_Ba_45(2)_10cm_300s.Spe",
        0:  "NaI_Ba_0_10cm_300s.Spe",
        45: "NaI_Ba_45(1)_10cm_300s.Spe",
        90: "NaI_Ba_90(1)_10cm_300s.Spe",},
        "Cs":{
        -90: "NaI_Cs_90(2)_10cm_180s.Spe",
       # -60: "NaI_Cs_60(2)_10cm_180s.Spe",
        -45: "Nal_CS_45(2)_10cm_180s.Spe",
        0:  "NaI_Cs_0_10cm_180s.Spe",
        45: "NaI_Cs_45(1)_10cm_180s.Spe",
        #60: "NaI_Cs_60(1)_10cm_180s.Spe",
        90: "NaI_Cs_90(1)_10cm_180s.Spe",},
        "Co":{
        -90: "NaI_Co_90(2)_10cm_420s.Spe",
        -45: "NaI_Co_45(2)_10cm_420s.Spe",
        0:  "NaI_Co_0_10cm_420s.Spe",
        45: "NaI_Co_45(1)_10cm_420s.Spe",
        90: "NaI_Co_90(1)_10cm_420s.Spe",},
        }},
    "CdTe": {
                "background": "CdTe_Background_300s.mca",
                "sources": {"Am":{
            -90: "CdTe_Am_90(2)_10cm_180s.mca",
            -45: "CdTe_Am_45(2)_10cm_180s.mca",
            0:  "CdTe_Am_0_10cm_180s.mca",
            45: "CdTe_Am_45(1)_10cm_180s.mca",
            90: "CdTe_Am_90(1)_10cm_180s.mca",},
            "Ba":{
            -90: "CdTe_Ba_90(2)_10cm_300s.mca",
            -45: "CdTe_Ba_45(2)_10cm_300s.mca",
            0:  "CdTe_Ba_0_10cm_300s.mca",
            45: "CdTe_Ba_45(1)_10cm_300s.mca",
            90: "CdTe_Ba_90(1)_10cm_300s.mca",},
            "Cs":{
            -90: "CdTe_Cs_90(2)_10cm_180s.mca",
            -45: "CdTe_Cs_45(2)_10cm_180s.mca",
            0:  "CdTe_Cs_0_10cm_180s.mca",
            45: "CdTe_Cs_45(1)_10cm_180s.mca",
            90: "CdTe_Cs_90(1)_10cm_180s.mca",},
            "Co":{
            -90: "CdTe_Co_90(2)_10cm_420s.mca",
            -45: "CdTe_Co_45(2)_10cm_420s.mca",
            0:  "CdTe_Co_0_10cm_420s.mca",
            45: "CdTe_Co_45(1)_10cm_420s.mca",
            90: "CdTe_Co_90(1)_10cm_420s.mca",},
            },
    
    }}


#A dictionary Of COunts
#
# Reference photon energies in keV, as specified by your original main.py.
# Select exactly the corresponding photopeaks in each interactive fit.
Known_Energies = {
    "Am": [59.5409],
    "Ba": [276.3989, 302.8508, 356.0129],
    "Cs": [661.657],
    "Co": [1173.228, 1332.492],
}


def make_detector(name):

    background_filename = Detectors[name]["background"]
    files = Detectors[name]["sources"]

    background_path = next(Path(".").rglob(background_filename))

    spectra = {}

    for source, filename in files.items():

        file_path = next(Path(".").rglob(filename))

        spectra[source] = Spectrum.from_file(
            file_path,
            background_file=background_path
        )

    return Detector(name, spectra, Known_Energies)

def make_detector_for_Offaxis(name, source):
    background_filename = measurements[name]["background"]
    files = measurements[name]["sources"][source]
    print("files to load:", files)

    background_path = next(Path(".").rglob(background_filename))

    spectra = {}

    for angle, filename in files.items():
        file_path = next(Path(".").rglob(filename))
        spectra[angle] = Spectrum.from_file(
            file_path,
            background_file=background_path
        )
        print("loaded angle", angle)

    print("returning angles:", list(spectra))
    return spectra





def main():
    all_r={}
    for name in Detectors:
        det = make_detector(name)
        det.calibrate()
        det.calculate_resolution()

        
        absolute_efficiency(det)
        plot_absolute_efficiency(det)

        for source in measurements[name]["sources"]:
            spectra = make_detector_for_Offaxis(name, source)
            for energy, r in run_off_axis(det, spectra, source).items():
                all_r[(name, source, energy)] = r

        plot_comparison(all_r)




if __name__ == "__main__":
    main()
