import qiskit
import numpy as np
from qiskit import transpile
import matplotlib.pyplot as plt
from qiskit_aer import Aer
import winsound
#variables modifiable par l'utilisateur
q = 0.05
input_note = [1, 2]
#fin des variables modifiables
n = len(input_note)
qr = qiskit.QuantumRegister(6)
cr = qiskit.ClassicalRegister(3 * n)
qc = qiskit.QuantumCircuit(qr, cr)

for i in range(n):
    if i % 2 == 0:
        sub = 0
    else:
        sub = 1
    qc.h([qr[0 + sub * 3], qr[1 + sub * 3], qr[2 + sub * 3]])
    qc.barrier()
    qc.rz(np.pi * 1 / 4  * input_note[i], qr[2 + sub * 3])
    qc.rz(np.pi * 1 / 2 * input_note[i], qr[1 + sub * 3])
    qc.rz(np.pi * 1 * input_note[i], qr[0 + sub * 3])
    qc.barrier()
    if i != 0:
        sub2 = (sub + 1) % 2
        for j in range(3):
            for _ in range(3 + j):
                qc.crz(q * np.pi / 4, qr[j + sub2 * 3], qr[0 + sub * 3])
            for _ in range(2 + j):
                qc.crz(q * np.pi / 4, qr[j + sub2 * 3], qr[1 + sub * 3])
            for _ in range(1 + j):
                qc.crz(q * np.pi / 4, qr[j + sub2 * 3], qr[2 + sub * 3])
        qc.barrier()
        qc.measure([qr[0 + sub2 * 3], qr[1 + sub2 * 3], qr[2 + sub2 * 3]],
                   [cr[0 + (i - 1) * 3], cr[1 + (i - 1) * 3], cr[2 + (i - 1) * 3]])
        qc.reset([qr[0 + sub2 * 3], qr[1 + sub2 * 3], qr[2 + sub2 * 3]])
    qc.barrier()
    qc.h(qr[0 + sub * 3])
    qc.crz(-np.pi * 2 / (2 ** 2), qr[0 + sub * 3], qr[1 + sub * 3])
    qc.crz(-np.pi * 2 / (2 ** 3), qr[0 + sub * 3], qr[2 + sub * 3])
    qc.h(qr[1 + sub * 3])
    qc.crz(-np.pi * 2 / (2 ** 2), qr[1 + sub * 3], qr[2 + sub * 3])
    qc.h(qr[2 + sub * 3])
    qc.swap(qr[0 + sub * 3], qr[2 + sub * 3])
    qc.barrier()
    if i == n - 1:
        qc.measure([qr[0 + sub * 3], qr[1 + sub * 3], qr[2 + sub * 3]],
                   [cr[0 + i * 3], cr[1 + i * 3], cr[2 + i * 3]])


def run(qc):
    backend = Aer.get_backend('qasm_simulator')
    newcircuit = transpile(qc, backend)
    result = backend.run(newcircuit, shots=1024).result()
    return result.get_counts(qc)


def parseur(result):
    return [result[i:i + 3][::-1] for i in range(0, len(result), 3)]

def play(tab_note):
    for i in tab_note:
        if i == 1:
            winsound.Beep(262, 500)
        elif i == 2:
            winsound.Beep(294, 500)
        elif i == 3:
            winsound.Beep(330, 500)
        elif i == 4:
            winsound.Beep(349, 500)
        elif i == 5:
            winsound.Beep(392, 500)
        elif i == 6:
            winsound.Beep(440, 500)
        elif i == 7:
            winsound.Beep(494, 500)
        else:
            print("Error")
            break



counts = run(qc)
measurement_result = list(counts.keys())[0].split()
measurement_result = ''.join(measurement_result)
plt.ioff()
measurement_result = parseur(measurement_result)
print(measurement_result, "measurement_result_binary")
measurement_result = [int(i, 2) for i in measurement_result]
print(measurement_result, "measurement_result_decimal")

# measurement_result.sort()
# print(measurement_result, "measurement_result")
fig = qc.draw("mpl", style="clifford")
fig.savefig('circuit.png')
plt.close(fig)
play(measurement_result)
