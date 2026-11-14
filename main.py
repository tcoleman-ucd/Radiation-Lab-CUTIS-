"""Load source spectra for BGO, NaI and CdTe and calibrate each detector."""

from pathlib import Path

from spectra import Spectrum
from detector import Detector



Detectors = {
    "BGO": {
        "radius": 0.0254,
        "background": "BGO_Background_180s.Spe",

        "distances": {
            "Am": 0.1,
            "Ba": 0.1,
            "Cs": 0.1,
            "Co": 0.1
        },

        "sources": {
            "Am": "BGO_Am_0_10cm_180s.Spe",
            "Ba": "BGO_Ba_0_10cm_300s.Spe",
            "Cs": "BGO_Cs_0_10cm_180s.Spe",
            "Co": "BGO_Co_0_10cm_420s.Spe"
        },

        "off_axis": {
            "Am": {
                -90: "BGO_Am_90(2)_10cm_180s.Spe",
                -45: "BGO_Am_45(2)_10cm_180s.Spe",
                45:  "BGO_Am_45(1)_10cm_180s.Spe",
                90:  "BGO_Am_90(1)_10cm_180s.Spe"
            },
            "Ba": {
                -90: "BGO_Ba_90(2)_10cm_300s.Spe",
                -45: "BGO_Ba_45(2)_10cm_300s.Spe",
                45:  "BGO_Ba_45(1)_10cm_300s.Spe",
                90:  "BGO_Ba_90(1)_10cm_300s.Spe"
            },
            "Cs": {
                -90: "BGO_Cs_90(2)_10cm_180s.Spe",
                -45: "BGO_Cs_45(2)_10cm_180s.Spe",
                45:  "BGO_Cs_45(1)_10cm_180s.Spe",
                90:  "BGO_Cs_90(1)_10cm_180s.Spe"
            },
            "Co": {
                -90: "BGO_Co_90(2)_10cm_420s.Spe",
                -45: "BGO_Co_45(2)_10cm_420s.Spe",
                45:  "BGO_Co_45(1)_10cm_420s.Spe",
                90:  "BGO_Co_90(1)_10cm_420s.Spe"
            }
        }
    },

    "NaI": {
        "radius": 0.02921,
        "background": "Nal_Background_420s.Spe",

        "distances": {
            "Am": 0.1,
            "Ba": 0.1,
            "Cs": 0.1,
            "Co": 0.1
        },

        "sources": {
            "Am": "NaI_Am_0_10cm_180s.Spe",
            "Ba": "NaI_Ba_0_10cm_300s.Spe",
            "Cs": "NaI_Cs_0_10cm_180s.Spe",
            "Co": "NaI_Co_0_10cm_420s.Spe"
        },

        "off_axis": {
            "Am": {
                -90: "NaI_Am_90(2)_10cm_180s.Spe",
                -45: "NaI_Am_45(2)_10cm_180s.Spe",
                45:  "NaI_Am_45(1)_10cm_180s.Spe",
                90:  "NaI_Am_90(1)_10cm_180s.Spe"
            },
            "Ba": {
                -90: "NaI_Ba_90(2)_10cm_300s.Spe",
                -45: "NaI_Ba_45(2)_10cm_300s.Spe",
                45:  "NaI_Ba_45(1)_10cm_300s.Spe",
                90:  "NaI_Ba_90(1)_10cm_300s.Spe"
            },
            "Cs": {
                -90: "NaI_Cs_90(2)_10cm_180s.Spe",
                -45: "Nal_Cs_45(2)_10cm_180s.Spe",
                45:  "NaI_Cs_45(1)_10cm_180s.Spe",
                90:  "NaI_Cs_90(1)_10cm_180s.Spe"
            },
            "Co": {
                -90: "NaI_Co_90(2)_10cm_420s.Spe",
                -45: "NaI_Co_45(2)_10cm_420s.Spe",
                45:  "NaI_Co_45(1)_10cm_420s.Spe",
                90:  "NaI_Co_90(1)_10cm_420s.Spe"
            }
        }
    },

    "CdTe": {
        "radius": 0.005,
        "background": "CdTe_Background_300s.mca",

        "distances": {
            "Am": 0.1,
            "Ba": 0.1,
            "Cs": 0.1,
            "Co": 0.1
        },

        "sources": {
            "Am": "CdTe_Am_0_10cm_180s.mca",
            "Ba": "CdTe_Ba_0_10cm_300s.mca",
            "Cs": "CdTe_Cs_10cm_180s.mca",
            "Co": "CdTe_Co_0_10cm_420s.mca"
        },

        "off_axis": {
            "Am": {
                -90: "CdTe_Am_90(2)_10cm_180s.mca",
                -45: "CdTe_Am_45(2)_10cm_180s.mca",
                45:  "CdTe_Am_45(1)_10cm_180s.mca",
                90:  "CdTe_Am_90(1)_10cm_180s.mca"
            },
            "Ba": {
                -90: "CdTe_Ba_90(2)_10cm_300s.mca",
                -45: "CdTe_Ba_45(2)_10cm_300s.mca",
                45:  "CdTe_Ba_45(1)_10cm_300s.mca",
                90:  "CdTe_Ba_90(1)_10cm_300s.mca"
            },
            "Cs": {
                -90: "CdTe_Cs_90(2)_10cm_180s.mca",
                -45: "CdTe_Cs_45(2)_10cm_180s.mca",
                45:  "CdTe_Cs_45(1)_10cm_180s.mca",
                90:  "CdTe_Cs_90(1)_10cm_180s.mca"
            },
            "Co": {
                -90: "CdTe_Co_90(2)_10cm_420s.mca",
                -45: "CdTe_Co_45(2)_10cm_420s.mca",
                45:  "CdTe_Co_45(1)_10cm_420s.mca",
                90:  "CdTe_Co_90(1)_10cm_420s.mca"
            }
        }
    }
}

# Reference photon energies in keV, as specified by the manual.
# Select exactly the corresponding photopeaks in each interactive fit.
Known_Energies = {
    "Am": [
        26.3446,
        33.1963,
        59.5409
    ],

    "Ba": [
        53.1622,
        79.6142,
        80.9979,
        276.3989,
        302.8508,
        356.0129,
        383.8485
    ],

    "Cs": [
        661.657
    ],

    "Co": [
        1173.228,
        1332.492
    ]
}




def make_detector(name):

    background_filename = Detectors[name]["background"]
    files = Detectors[name]["sources"]

    background_path = next(Path(".").rglob(background_filename))

    spectra = {}

    # Load on-axis spectra
    for source, filename in files.items():

        file_path = next(Path(".").rglob(filename))

        spectra[source] = Spectrum.from_file(
            file_path,
            background_file=background_path,
            distance=Detectors[name]["distances"][source]
        )

    # Load off-axis spectra
    off_axis_spectra = {}

    for source, angles in Detectors[name]["off_axis"].items():

        # Reuse the calibration spectrum at 0 degrees
        off_axis_spectra[source] = {0: spectra[source]}

        for angle, filename in angles.items():

            file_path = next(Path(".").rglob(filename))

            off_axis_spectra[source][angle] = Spectrum.from_file(
                file_path,
                background_file=background_path,
                distance=Detectors[name]["distances"][source]
            )

    return Detector(
        name,
        spectra,
        Known_Energies,
        radius=Detectors[name]["radius"],
        off_axis_spectra=off_axis_spectra
    )



def main():
    # for name in ["BGO", "NaI", "CdTe"]:
    #     detector = make_detector(name)
    #     detector.calibrate()


    bgo = make_detector("BGO")
    for angle, spectrum in sorted(bgo.off_axis_spectra["Co"].items()):
        print(f"Co spectrum at {angle}°")
        spectrum.plot(corrected=False)

    # bgo.calibrate()



    # for angle in [0, 90]:
    #     spectrum = bgo.off_axis_spectra["Cs"][angle]
    #
    #     print(
    #         angle,
    #         "degrees:",
    #         spectrum.corrected_cps().sum(),
    #         "total counts/s"
    #     )

    #
    # # bgo.calibrate(identification='terminal')
    # bgo.calibrate()
    # # bgo.calculate_absolute_efficiency()
    # # bgo.calculate_resolution()
    #
    # bgo.calculate_absolute_efficiency(plot=True)
    #
    # bgo.calculate_intrinsic_efficiency(plot=True)
    # bgo.calculate_off_axis("Co")
    # for source in bgo.off_axis_spectra:
    #     bgo.calculate_off_axis(source)


if __name__ == "__main__":
    main()
