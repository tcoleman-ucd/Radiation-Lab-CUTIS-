"""Load source spectra for BGO, NaI and CdTe and calibrate each detector."""
import difflib
from pathlib import Path

from spectra import Spectrum
from detector import Detector
from Efficiency import fit_distance_peaks, absolute_efficiency, intrinsic_eff
import argparse

def find_file(filename):
    """Find a file under the folder containing this script, with a clear error."""
    root = Path(__file__).resolve().parent
    matches = list(root.rglob(filename))
    if matches:
        return matches[0]

    all_files = [p.name for p in root.rglob("*") if p.is_file()]
    close = difflib.get_close_matches(filename, all_files, n=5, cutoff=0.5)
    raise FileNotFoundError(
        f"\nLooking for: {filename!r}"
        f"\nClosest real files: {close}"
    )

Detectors = {
    "BGO": {
        "radius": 0.0254,
        "background": "BGO_Background_180s.Spe",

        "distances": {
            0.1: {
                "Am": "BGO_Am_0_10cm_180s.Spe",
                "Ba": "BGO_Ba_0_10cm_300s.Spe",
                "Cs": "BGO_Cs_0_10cm_180s.Spe",
                "Co": "BGO_Co_0_10cm_420s.Spe"
            },
            0.2: {
                "Am": "BGO_Am_0_20cm_180s.Spe",
                "Ba": "BGO_Ba_0_20cm_300s.Spe",
                "Cs": "BGO_Cs_0_20cm_180s.Spe",
                "Co": "BGO_Co_0_20cm_420s.Spe"
            }
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
        "radius": 0.0254,
        "background": "Nal_Background_420s.Spe",

        "distances": {
            0.1: {
                "Am": "NaI_Am_0_10cm_180s.Spe",
                "Ba": "NaI_Ba_0_10cm_300s.Spe",
                "Cs": "NaI_Cs_0_10cm_180s.Spe",
                "Co": "NaI_Co_0_10cm_420s.Spe"
            },
            0.2: {
                "Am": "NaI_Am_0_20cm_180s.Spe",
                "Ba": "NaI_Ba_0_20cm_300s.Spe",
                "Cs": "NaI_Cs_0_20cm_180s.Spe",
                "Co": "NaI_Co_0_20cm_420s.Spe"
            }
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
                45:  "Nal_Cs_45(1)_10cm_180s.Spe",
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
        0.1: {
            "Am": "CdTe_Am_0_10cm_180s.mca",
            "Ba": "CdTe_Ba_0_10cm_300s.mca",
            "Cs": "CdTe_Cs_0_10cm_180s.mca",
            "Co": "CdTe_Co_0_10cm_420s.mca"
        },
        0.2: {
            "Am": "CdTe_Am_0_20cm_180s.mca",
            "Ba": "CdTe_Ba_0_20cm_300s.mca",
            "Cs": "CdTe_Cs_0_20cm_180s.mca",
            "Co": "CdTe_Co_0_20cm_420s.mca"
        }
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

    # Use the first distance for calibration (10 cm)
    reference_distance = next(iter(Detectors[name]["distances"]))
    files = Detectors[name]["distances"][reference_distance]

    background_path = next(Path(".").rglob(background_filename))

    spectra = {}

    # Load on-axis calibration spectra
    for source, filename in files.items():

        file_path = next(Path(".").rglob(filename))

        spectra[source] = Spectrum.from_file(
            file_path,
            background_file=background_path,
            distance=reference_distance
        )

    # Load additional distance spectra
    distance_spectra = {}

    for distance, files in Detectors[name]["distances"].items():

        if distance == reference_distance:
            continue

        distance_spectra[distance] = {}

        for source, filename in files.items():

            file_path = find_file(filename)

            distance_spectra[distance][source] = Spectrum.from_file(
                file_path,
                background_file=background_path,
                distance=distance
            )

    # Load off-axis spectra
    off_axis_spectra = {}

    for source, angles in Detectors[name]["off_axis"].items():

        # Reuse the calibration spectrum at 0 degrees
        off_axis_spectra[source] = {0: spectra[source]}

        for angle, filename in angles.items():

            file_path = find_file(filename)

            off_axis_spectra[source][angle] = Spectrum.from_file(
                file_path,
                background_file=background_path,
                distance=reference_distance
            )

    return Detector(
        name,
        spectra,
        Known_Energies,
        radius=Detectors[name]["radius"],
        off_axis_spectra=off_axis_spectra,
        distance_spectra=distance_spectra
    )





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("Calibration", nargs ="+")
    parser.add_argument("--skip-off-axis", action="store_true")

    args = parser.parse_args()
  
    all_r={}
    if len(args.Calibration) > 1:
        for name in args.Calibration:
            source_keys = list(Detectors[name]["distances"][0.1].keys())
            det = make_detector(name)
            det.calibrate()
        
            det.calculate_resolution()
            det.calculate_absolute_efficiency()
            det.calculate_intrinsic_efficiency()

            if not args.skip_off_axis:
                for source in source_keys:
                    det.calculate_off_axis(source)


  
    else:
            name = args.Calibration[0]
            source_keys = list(Detectors[name]["distances"][0.1].keys())
            det = make_detector(name)
            det.calibrate()
            det.calculate_resolution()
            det.calculate_absolute_efficiency()
            det.calculate_intrinsic_efficiency()
            if not args.skip_off_axis:
                for source in source_keys:
                    det.calculate_off_axis(source)
            
        




    # bgo = make_detector("NaI")



  

    # print(bgo.calculate_absolute_efficiency())

    # print(bgo.calculate_intrinsic_efficiency())



if __name__ == "__main__":
    main()
