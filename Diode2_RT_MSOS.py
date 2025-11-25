import numpy as np
import matplotlib.pyplot as plt

# Messdaten 30 Grad Celsius
Spannung_U_RT_MS = [0.0, 0.03, 0.06, 0.09, 0.12, 0.16, 0.19, 0.22,
             0.25, 0.28, 0.31, 0.34, 0.38, 0.41, 0.44, 0.47, 0.5, 0.52,
             0.56, 0.59, 0.62,  0.66, 0.69, 0.72, 0.75, 0.78, 0.81, 0.84,
             0.88, 0.91, 0.94, 0.97, 1.0]

DC_1_RT_MS = [ -4e-05, -5e-05, -1e-05, -8e-05, -6e-05, -7e-05, -0.00014, -8e-05,
    -4e-05, -2e-05, -1e-05, -7e-05, -8e-05, -1e-05, 1e-05, 9e-05,
    0.00014, 0.00036, 0.00057, 0.00115, 0.0021, 0.00553, 0.01062, 0.02046,
    0.03975, 0.07553, 0.13733, 0.23353, 0.40738, 0.59114, 0.80888,
    1.05551, 1.32704]

Spannung_U_RT_OS =[ 0.0, 0.03, 0.06, 0.09, 0.12, 0.16, 0.19, 0.22,
    0.25, 0.28, 0.31, 0.34, 0.38, 0.41, 0.44, 0.47,
    0.5, 0.53, 0.56, 0.59, 0.62, 0.66, 0.69, 0.72,
    0.75, 0.78, 0.81, 0.84, 0.88, 0.91, 0.94, 0.97,
    1.0]


DC_1_RT_OS =[-9e-05, -7e-05, -0.00011, -5e-05, -0.00011, -9e-05, 0.0, -4e-05,
    -5e-05, 0.0, -7e-05, -6e-05, -0.0001, -3e-05, -0.0, 6e-05,
    0.00014, 0.00035, 0.00058, 0.00114, 0.00214, 0.00538, 0.00965,
    0.01711, 0.02922, 0.04673, 0.06948, 0.09695, 0.13644, 0.17082,
    0.20572, 0.23903, 0.23122]


plt.figure(figsize=(10, 6))
plt.plot(Spannung_U_RT_MS, DC_1_RT_MS, label="Mit Sense", marker='o', linestyle='-', color='b')

plt.plot(Spannung_U_RT_OS, DC_1_RT_OS, label="Ohne Sense", marker='o', linestyle='-', color='r')
plt.legend()
plt.title('Messdaten der 2 Diode 30°C')
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
plt.title("Logarithmische Darstellung – 2. Diode bei 30°C")
plt.show()
