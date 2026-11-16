
import os
import matplotlib.pyplot as plt

def plot():
    plt.figure(figsize=(10, 6))
    plt.errorbar(E, R*100, yerr=R_err*100, fmt="o", color="black", capsize=3, label="Data")
    offsets = [(-10, 28), (10, -32)]      # first label above-left, second below-right
    for i, pk in enumerate(peaks):
        det, iso = Path(pk["source"]).stem.split("_")[:2]
        plt.annotate(f"{iso} {E[i]:.0f} keV", (E[i], R[i] * 100),
                        textcoords="offset points", xytext=offsets[i % 2],
                        ha="center", fontsize=9,
                        arrowprops=dict(arrowstyle="-", color="gray", lw=0.8))
    plt.plot(E_line, R_line*100, "r-", lw=2, label=r"Fit: $R=\sqrt{aE^{-2}+bE^{-1}+c}$")
    plt.xscale("log")
    plt.yscale("log")
    #plt.xticks([10, 20, 50, 100, 200, 500, 1000, 2000], ["10", "20", "50", "100", "200", "500", "1000", "2000"])
    #plt.yticks([1, 2, 5, 10, 20, 50, 60, 70, 80, 90, 100], ["1", "2", "5", "10", "20", "50", "60","70", "80", "90", "100"])
    plt.xlabel("Energy (keV)"); plt.ylabel("Resolution  ΔE/E (%)")
    plt.title(title)
    plt.grid(True, which="both", ls="--", alpha=0.5); plt.legend()
    file_path = os.path.join(Plots, "plot_1.jpg")
    plt.savefig("plot", format = "jpg", dpi=300, bbox_inches = "tight")
    plt.close()