import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
# Messdaten 30 Grad Celsius
Spannung_U_RT_MS = [0.0, 0.03, 0.06, 0.09, 0.12, 0.16, 0.19, 0.22,
             0.25, 0.28, 0.31, 0.34, 0.38, 0.41, 0.44, 0.47, 0.5, 0.52,
             0.56, 0.59, 0.62,  0.66, 0.69, 0.72, 0.75, 0.78, 0.81, 0.84,
             0.88, 0.91, 0.94, 0.97, 1.0]

DC_1_RT_MS = [-0.00015,-0.0001,-3e-05,-4e-05,-9e-05,-1e-05,-9e-05,2e-05,-0.00011,
        -6e-05,-0.0,-4e-05,-1e-05,-2e-05,-1e-05,1e-05,9e-05,0.00015,0.00042,
        0.00089,0.00186,0.0054,0.01081,0.02164,0.0431,0.08286,
        0.15039,0.253,0.43717,0.63489,0.86762,1.13272,1.42627,]

Spannung_U_RT_OS =[ 0.0, 0.03, 0.06, 0.09, 0.12, 0.16, 0.19, 0.22,
    0.25, 0.28, 0.31, 0.34, 0.38, 0.41, 0.44, 0.47,
    0.5, 0.53, 0.56, 0.59, 0.62, 0.66, 0.69, 0.72,
    0.75, 0.78, 0.81, 0.84, 0.88, 0.91, 0.94, 0.97,
    1.0]


DC_1_RT_OS =[ -8e-05, -5e-05, -6e-05, -8e-05, -9e-05, -7e-05, -5e-05, -0.00012,
    -0.00012, 6e-05, -7e-05, -0.00011, -0.00012, -0.00012, -0.00015, -0.0,
    2e-05, 0.00014, 0.00039, 0.00091, 0.00183, 0.00511, 0.00989, 0.01807,
    0.0313, 0.05008, 0.0741, 0.10284, 0.14365, 0.17875, 0.2109, 0.24082,
    0.23282 ]


plt.figure(figsize=(10, 6))
plt.plot(Spannung_U_RT_MS, DC_1_RT_MS, label="Mit Sense", marker='o', linestyle='-', color='b')

plt.plot(Spannung_U_RT_OS, DC_1_RT_OS, label="Ohne Sense", marker='o', linestyle='-', color='r')
plt.legend()
plt.title('Messdaten der 1 Diode 30°C')
plt.xlabel('Spannung (V)')
plt.ylabel('Strom (A)')
plt.grid()


plt.show()

U_ms = np.array(Spannung_U_RT_MS)
I_ms = np.array(DC_1_RT_MS)

U_os = np.array(Spannung_U_RT_OS)
I_os = np.array(DC_1_RT_OS)

# 10 pA Offset laut Praktikumsskript
offset = 1e-11



# Betrag + Offset
I_ms_log = np.abs(I_ms) + offset
I_os_log = np.abs(I_os) + offset

plt.figure(figsize=(20, 10))

plt.semilogy(U_ms, I_ms_log, 'o-', label="Mit Sense")
plt.semilogy(U_os, I_os_log, 'o-', label="Ohne Sense")
plt.legend()
plt.grid(True, which="both")
plt.xlabel("Spannung (V)")
plt.ylabel("|Strom| + 10 pA (A)")
plt.title("Logarithmische Darstellung – 1. Diode bei 30°C")
plt.show()

lnI = np.log(I_ms_log)
mask = U_ms > 0.45   # Beispiel: nur Punkte ab 0.45 V
m, b = np.polyfit(U_ms[mask], lnI[mask], 1)

U_fit = np.linspace(0.4, 0.8, 300)  # typischer Shockley-Bereich
I_fit = np.exp(m * U_fit + b)

plt.semilogy(U_fit, I_fit, '-', color='black', linewidth=2, label="Shockley-Gerade")
plt.legend()
plt.show()
