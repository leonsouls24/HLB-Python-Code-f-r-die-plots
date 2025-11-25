from matplotlib.backends.backend_pdf import PdfPages

# Matplotlib-Style (ohne echtes LaTeX, aber mit Mathtext)
mpl.rcParams.update({
    "text.usetex": False,  # wichtig: kein pdflatex nötig
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "axes.labelsize": 9,
    "font.size": 9,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "figure.dpi": 300,
})

OFFSET = 1e-11   # 10 pA Offset für Log-Plot


# ----------------------------------------
# Hilfsfunktionen
# ----------------------------------------
def load_diode_csv(path: Path):
    """CSV mit Spalten 'Variable' und 'DC I' laden, Hochkommas entfernen, sortieren."""
    df = pd.read_csv(path, sep=";", engine="python")
    U = df["Variable"].astype(str).str.replace("'", "", regex=False).astype(float).to_numpy()
    I = df["DC I"].astype(str).str.replace("'", "", regex=False).astype(float).to_numpy()
    order = np.argsort(U)
    return U[order], I[order]


def find_best_linear_window_log(U, logI, window_points=10):
    """
    Finde den am stärksten linear korrelierten Bereich in log(I) vs. U (Region 2).
    """
    n = len(U)
    if n < window_points:
        return 0, n

    best_r = -np.inf
    best_i0 = 0
    for i0 in range(0, n - window_points):
        i1 = i0 + window_points
        U_win = U[i0:i1]
        Y_win = logI[i0:i1]
        if np.any(~np.isfinite(Y_win)):
            continue
        if np.all(np.diff(U_win) <= 0):
            continue
        if np.std(U_win) == 0 or np.std(Y_win) == 0:
            continue
        r = abs(np.corrcoef(U_win, Y_win)[0, 1])
        if r > best_r:
            best_r = r
            best_i0 = i0
    return best_i0, best_i0 + window_points


def find_best_linear_window_IV(U, I, window_points=8):
    """
    Finde den quasi-ohmschen Bereich (Region 3) in I vs. U:
    linear und eher bei hohen Spannungen.
    """
    n = len(U)
    if n < window_points:
        return 0, n

    best_score = -np.inf
    best_i0 = 0
    U_max = np.max(U)

    for i0 in range(0, n - window_points):
        i1 = i0 + window_points
        U_win = U[i0:i1]
        Y_win = I[i0:i1]

        if np.any(~np.isfinite(Y_win)):
            continue
        if np.all(np.diff(U_win) <= 0):
            continue
        if np.std(U_win) == 0 or np.std(Y_win) == 0:
            continue

        r = abs(np.corrcoef(U_win, Y_win)[0, 1])
        meanU = np.mean(U_win)

        # Score: gute Linearität + bevorzugt hohe Spannungen
        score = r + 0.3 * (meanU / U_max)
        if score > best_score:
            best_score = score
            best_i0 = i0

    return best_i0, best_i0 + window_points


def analyze_log_region(U, I, offset=OFFSET, window_points=10):
    """
    Analyse des halb-logarithmischen Plots:
    - Region 2 (Shockley-Bereich) finden
    - lineare Regression in log10(I_eff)
    - I0 bestimmen
    - Grenzen U1, U2 der Region 2 liefern
    """
    mask = U > 0
    U_fwd = U[mask]
    I_fwd = I[mask]

    I_eff = np.abs(I_fwd) + offset
    logI = np.log10(I_eff)

    i0, i1 = find_best_linear_window_log(U_fwd, logI, window_points=window_points)
    U_fit = U_fwd[i0:i1]
    logI_fit = logI[i0:i1]

    if len(U_fit) < 2:
        # fallback
        return U_fwd, I_eff, None, None, None, None, None

    # log10(I_eff) = m * U + c
    m, c = np.polyfit(U_fit, logI_fit, 1)
    I0_plot = 10 ** c  # I_eff bei U=0
    I0 = max(I0_plot - offset, 0.0)

    U1 = float(U_fit[0])
    U2 = float(U_fit[-1])

    return U_fwd, I_eff, m, c, I0, U1, U2


def analyze_linear_region(U, I, window_points=8):
    """
    Analyse der linearen Diodenkennlinie im ohmschen Bereich (Region 3):
    - quasi-linearer Bereich finden
    - I = a*U + b fitten
    - daraus U_F und R_i bestimmen
    """
    mask = U > 0
    U_fwd = U[mask]
    I_fwd = I[mask]

    if len(U_fwd) < 2:
        return U_fwd, I_fwd, None, None, None, None, None

    i0, i1 = find_best_linear_window_IV(U_fwd, I_fwd, window_points=window_points)
    U_fit = U_fwd[i0:i1]
    I_fit = I_fwd[i0:i1]

    if len(U_fit) < 2:
        return U_fwd, I_fwd, None, None, None, None, None

    # I = a*U + b
    a, b = np.polyfit(U_fit, I_fit, 1)

    if a <= 0:
        R_i = np.nan
        U_F = np.nan
    else:
        R_i = 1.0 / a
        U_F = -b / a

    return U_fwd, I_fwd, a, b, R_i, U_F, (U_fit[0], U_fit[-1])


# ----------------------------------------
# Hauptskript
# ----------------------------------------


def main():

    # *** Fester Pfad zu deinen Messdaten ***
    base_dir = Path(r"C:\Users\Liod\Desktop\HLB Labor\Protokoll\MessDiode#")

    # Reihenfolge: Sense, RT, 60, 90 für Diode 1 und 2
    datasets = [
        {"filename": "Werte_D1_RT_MS.csv", "diode": 1, "temp_label": r"Sense Raumtemperatur"},
        {"filename": "Werte_D1_RT_OS.csv", "diode": 1, "temp_label": r"Raumtemperatur"},
        {"filename": "Werte_D1_60_MS.csv", "diode": 1, "temp_label": r"60$^\circ$C"},
        {"filename": "Werte_D1_90_MS.csv", "diode": 1, "temp_label": r"90$^\circ$C"},
        {"filename": "Werte_D2_RT_MS.csv", "diode": 2, "temp_label": r"Sense Raumtemperatur"},
        {"filename": "Werte_D2_RT_OS.csv", "diode": 2, "temp_label": r"Raumtemperatur"},
        {"filename": "Werte_D2_60_MS.csv", "diode": 2, "temp_label": r"60$^\circ$C"},
        {"filename": "Werte_D2_90_MS.csv", "diode": 2, "temp_label": r"90$^\circ$C"},
    ]


    pdf_path = Path("Diodenkennlinien.pdf")
    with PdfPages(pdf_path) as pdf:

        # ----------------------------
        # Lineare Kennlinien – Diode 1
        # ----------------------------
        fig_lin1, axes_lin1 = plt.subplots(
            4, 1, figsize=(8.27, 11.69), constrained_layout=True
        )
        axes_lin1 = np.atleast_1d(axes_lin1)

        for ax, ds in zip(axes_lin1, datasets[:4]):
            path = base_dir / ds["filename"]
            U, I = load_diode_csv(path)

            U_fwd, I_fwd, a, b, R_i, U_F, (U_fit_min, U_fit_max) = analyze_linear_region(U, I)

            ax.plot(U, I, marker="o", linestyle="-", linewidth=0.8, markersize=2)

            ax.set_xlabel(r"$U_\mathrm{D}\,/\,\mathrm{V}$")
            ax.set_ylabel(r"$I_\mathrm{D}\,/\,\mathrm{A}$")

            title = rf"Diodenkennlinie - Diode {ds['diode']}, {ds['temp_label']}"
            ax.set_title(title)

            # U_F als vertikale gestrichelte Linie (von x-Achse bis Kurve)
            if U_F is not None and np.isfinite(U_F):
                y_min, y_max = ax.get_ylim()
                y_bottom = y_min
                y_top = np.interp(U_F, U_fwd, I_fwd,
                                  left=y_min, right=y_max)
                ax.vlines(U_F, y_bottom, y_top,
                          linestyles="dashed", linewidth=0.8)

            # Legende mit U_F und R_i
            if R_i is not None and np.isfinite(R_i) and U_F is not None and np.isfinite(U_F):
                UF_mV = U_F * 1e3
                label = (rf"$U_F \approx {UF_mV:.0f}\,\mathrm{{mV}},\ "
                         rf"R_i \approx {R_i:.2e}\,\Omega$")
                ax.legend([label], frameon=False, loc="best")

       # fig_lin1.suptitle(
        #    r"Lineare Diodenkennlinien für Diode 1. "
         #   r"$U_F$: Flussspannung, $R_i$: Serienwiderstand.",
          #  y=1.02
        #)
        pdf.savefig(fig_lin1)
        plt.close(fig_lin1)

        # ----------------------------
        # Lineare Kennlinien – Diode 2
        # ----------------------------
        fig_lin2, axes_lin2 = plt.subplots(
            4, 1, figsize=(8.27, 11.69), constrained_layout=True
        )
        axes_lin2 = np.atleast_1d(axes_lin2)

        for ax, ds in zip(axes_lin2, datasets[4:]):
            path = base_dir / ds["filename"]
            U, I = load_diode_csv(path)

            U_fwd, I_fwd, a, b, R_i, U_F, (U_fit_min, U_fit_max) = analyze_linear_region(U, I)

            ax.plot(U, I, marker="o", linestyle="-", linewidth=0.8, markersize=2)

            ax.set_xlabel(r"$U_\mathrm{D}\,/\,\mathrm{V}$")
            ax.set_ylabel(r"$I_\mathrm{D}\,/\,\mathrm{A}$")

            title = rf"Diodenkennlinie - Diode {ds['diode']}, {ds['temp_label']}"
            ax.set_title(title)

            # U_F als vertikale gestrichelte Linie (von x-Achse bis Kurve)
            if U_F is not None and np.isfinite(U_F):
                y_min, y_max = ax.get_ylim()
                y_bottom = y_min
                y_top = np.interp(U_F, U_fwd, I_fwd,
                                  left=y_min, right=y_max)
                ax.vlines(U_F, y_bottom, y_top,
                          linestyles="dashed", linewidth=0.8)

            # Legende mit U_F und R_i
            if R_i is not None and np.isfinite(R_i) and U_F is not None and np.isfinite(U_F):
                UF_mV = U_F * 1e3
                label = (rf"$U_F \approx {UF_mV:.0f}\,\mathrm{{mV}},\ "
                         rf"R_i \approx {R_i:.2e}\,\Omega$")
                ax.legend([label], frameon=False, loc="best")

       # fig_lin2.suptitle(
        #    r"Lineare Diodenkennlinien – Diode 2. "
        #    r"$U_F$: Flussspannung, $R_i$: Serienwiderstand.",
        #    y=0.97
        #)
        pdf.savefig(fig_lin2)
        plt.close(fig_lin2)

        # ----------------------------
        # Halb-logarithmische Kennlinien – Diode 1
        # ----------------------------
        fig_log1, axes_log1 = plt.subplots(
            4, 1, figsize=(8.27, 11.69), constrained_layout=True
        )
        axes_log1 = np.atleast_1d(axes_log1)

        for ax, ds in zip(axes_log1, datasets[:4]):
            path = base_dir / ds["filename"]
            U, I = load_diode_csv(path)

            U_fwd, I_eff, m, c, I0, U1, U2 = analyze_log_region(U, I, offset=OFFSET, window_points=10)

            if len(U_fwd) == 0:
                continue

            ax.semilogy(U_fwd, I_eff, marker="o", linestyle="-", linewidth=0.8, markersize=2)

            ax.set_xlabel(r"$U_\mathrm{D}\,/\,\mathrm{V}$")
            ax.set_ylabel(r"$|I_\mathrm{D}| + 10\,\mathrm{pA}/\mathrm{A}$")

            title = rf"Diodenkennlinie (halb-log.) -- Diode {ds['diode']}, {ds['temp_label']}"
            ax.set_title(title)

            if m is not None and c is not None and U1 is not None and U2 is not None:
                U_fit_line = np.linspace(U1, U2, 100)
                logI_fit_line = m * U_fit_line + c
                I_fit_line = 10 ** logI_fit_line
                ax.semilogy(U_fit_line, I_fit_line, linestyle="--", linewidth=0.8)

                y_min, y_max = ax.get_ylim()
                ax.vlines(U1, y_min, y_max, linestyles="dashed", linewidth=0.6)
                ax.vlines(U2, y_min, y_max, linestyles="dashed", linewidth=0.6)

                x1 = U1 * 0.7
                x2 = 0.5 * (U1 + U2)
                x3 = U2 + 0.05 * (U_fwd.max() - U_fwd.min())
                y_label = np.sqrt(y_min * y_max)

                ax.text(x1, y_label, r"1", ha="center", va="center")
                ax.text(x2, y_label, r"2", ha="center", va="center")
                ax.text(x3, y_label, r"3", ha="center", va="center")

                if I0 is not None and I0 > 0:
                    label = rf"$I_0 \approx {I0:.2e}\,\mathrm{{A}}$"
                    ax.legend([label], frameon=False, loc="best")

#        fig_log1.suptitle(
 #           r"Halb-logarithmische Kennlinien – Diode 1. "
  #          r"Bereiche: 1 = Rekombination in RLZ, "
   #         r"2 = Shockley-Bereich, "
    #        r"3 = Bahnwiderstand + starke Injektion.",
     #       y=0.97
      #  )
        pdf.savefig(fig_log1)
        plt.close(fig_log1)

        # ----------------------------
        # Halb-logarithmische Kennlinien – Diode 2
        # ----------------------------
        fig_log2, axes_log2 = plt.subplots(
            4, 1, figsize=(8.27, 11.69), constrained_layout=True
        )
        axes_log2 = np.atleast_1d(axes_log2)

        for ax, ds in zip(axes_log2, datasets[4:]):
            path = base_dir / ds["filename"]
            U, I = load_diode_csv(path)

            U_fwd, I_eff, m, c, I0, U1, U2 = analyze_log_region(U, I, offset=OFFSET, window_points=10)

            if len(U_fwd) == 0:
                continue

            ax.semilogy(U_fwd, I_eff, marker="o", linestyle="-", linewidth=0.8, markersize=2)

            ax.set_xlabel(r"$U_\mathrm{D}\,/\,\mathrm{V}$")
            ax.set_ylabel(r"$|I_\mathrm{D}| + 10\,\mathrm{pA}/\mathrm{A}$")

            title = rf"Diodenkennlinie (halb-log.) - Diode {ds['diode']}, {ds['temp_label']}"
            ax.set_title(title)

            if m is not None and c is not None and U1 is not None and U2 is not None:
                U_fit_line = np.linspace(U1, U2, 100)
                logI_fit_line = m * U_fit_line + c
                I_fit_line = 10 ** logI_fit_line
                ax.semilogy(U_fit_line, I_fit_line, linestyle="--", linewidth=0.8)

                y_min, y_max = ax.get_ylim()
                ax.vlines(U1, y_min, y_max, linestyles="dashed", linewidth=0.6)
                ax.vlines(U2, y_min, y_max, linestyles="dashed", linewidth=0.6)

                x1 = U1 * 0.7
                x2 = 0.5 * (U1 + U2)
                x3 = U2 + 0.05 * (U_fwd.max() - U_fwd.min())
                y_label = np.sqrt(y_min * y_max)

                ax.text(x1, y_label, r"1", ha="center", va="center")
                ax.text(x2, y_label, r"2", ha="center", va="center")
                ax.text(x3, y_label, r"3", ha="center", va="center")

                if I0 is not None and I0 > 0:
                    label = rf"$I_0 \approx {I0:.2e}\,\mathrm{{A}}$"
                    ax.legend([label], frameon=False, loc="best")

  #      fig_log2.suptitle(
  #          r"Halb-logarithmische Kennlinien – Diode 2. "
   #         r"Bereiche: 1 = Rekombination in RLZ, "
    #        r"2 = Shockley-Bereich, "
     #       r"3 = Bahnwiderstand + starke Injektion.",
      #      y=0.97
       # )
        pdf.savefig(fig_log2)
        plt.close(fig_log2)

    print(f"Fertig! PDF gespeichert als: {pdf_path}")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
