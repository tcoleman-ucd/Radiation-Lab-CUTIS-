"""Load source spectra for BGO, NaI and CdTe and calibrate each detector."""

from pathlib import Path

from spectra import Spectrum
from detector import Detector
from Efficiency import absolute_efficiency, plot_absolute_efficiency





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


def main():
    # for name in ["BGO", "NaI", "CdTe"]:
    #     detector = make_detector(name)
    #     detector.calibrate()

    bgo = make_detector("BGO")
    
    bgo.calibrate()

    bgo.calculate_resolution()
    
    absolute_efficiency(bgo)
    plot_absolute_efficiency(bgo)

if __name__ == "__main__":
    main()
