import qiskit
import numpy as np
from qiskit_aer import *
from qiskit import QuantumCircuit, transpile
from qiskit.visualization import plot_histogram
from qiskit.quantum_info import Statevector
import sounddevice as sd
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
from random import randint

qr = qiskit.QuantumRegister(5)
cr = qiskit.ClassicalRegister(3)
circuit = QuantumCircuit(qr, cr)
valeurstart = 1
n = 4
notes = []
lin = True


def plot_value_probabilities(value_probabilities):
    """
    Affiche un graphique des probabilités de chaque valeur (note) en pourcentage.

    :param value_probabilities: Dictionnaire contenant les probabilités des valeurs (notes) en pourcentage.
    """
    # Extraction des clés (valeurs) et des probabilités (en pourcentage)
    values = list(value_probabilities.keys())
    probabilities = list(value_probabilities.values())

    # Création du graphique
    plt.figure(figsize=(10, 6))
    plt.bar(values, probabilities, color='skyblue', edgecolor='black')

    # Ajout des étiquettes et du titre
    plt.xlabel("Valeurs (Notes)", fontsize=12)
    plt.ylabel("Probabilités (%)", fontsize=12)
    plt.title("Probabilités des valeurs mesurées", fontsize=14)
    plt.xticks(values, fontsize=10)
    plt.ylim(0, 100)

    # Affichage des pourcentages au-dessus des barres
    for i, prob in enumerate(probabilities):
        plt.text(values[i], prob + 1, f"{prob:.1f}%", ha='center', fontsize=10)

    # Affichage du graphique
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def addtionneur(circuit):
    circuit.ccx(qr[1], qr[3], qr[4])
    circuit.ccx(qr[1], qr[3], qr[2])
    circuit.ccx(qr[0], qr[3], qr[1])
    circuit.cx(qr[3], qr[0])
    circuit.ccx(qr[1], qr[4], qr[2])
    circuit.ccx(qr[0], qr[3], qr[4])
    circuit.ccx(qr[1], qr[3], qr[4])
    circuit.cx(qr[3], qr[4])


def soustracteur(circuit):
    for i in range(2):
        circuit.cx(qr[3], qr[0])
        circuit.ccx(qr[0], qr[1], qr[4])
        circuit.ccx(qr[4], qr[3], qr[2])
        circuit.ccx(qr[0], qr[1], qr[4])

    circuit.ccx(qr[0], qr[3], qr[1])

    circuit.ccx(qr[0], qr[1], qr[4])
    circuit.ccx(qr[4], qr[3], qr[2])

    circuit.ccx(qr[0], qr[1], qr[4])
    circuit.cx(qr[3], qr[2])
    circuit.cx(qr[3], qr[1])
    circuit.cx(qr[3], qr[0])


def initialisation(valeurstart, circuit):
    valeurstart = bin(valeurstart)[2:].zfill(3)
    for i in range(3):
        if valeurstart[i] == '1':
            circuit.x(qr[3 - i - 1])


def generate_note(frequency, duration=1.0, sample_rate=44100, amplitude=0.5):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = amplitude * np.sin(2 * np.pi * frequency * t)
    return wave


def map_note_to_frequency(note):
    # Map the note value to a frequency in the range of 220 Hz to 880 Hz (A3 to A5)
    base_frequency = 220  # A3
    frequency_range = 660  # A5 - A3
    max_note_value = 7  # Assuming note values range from 0 to 7
    frequency = base_frequency + (note / max_note_value) * frequency_range
    return frequency


initialisation(valeurstart, circuit)
last=5
max_rep=6
rep=1
lastbin=bin(last)[2:].zfill(5)
for i in lastbin:
    if i == '1':
        circuit.x(qr[5 - lastbin.index(i) - 1])
for i in range(n):
    for _ in range(randint(0,rep)):
        circuit.h(qr[3])
        circuit.barrier()
        addtionneur(circuit)
        circuit.barrier()
        circuit.x(qr[3])
        circuit.barrier()
        soustracteur(circuit)


        # le h magique
        # circuit.h(qr[3])
        circuit.barrier()
        if rep<max_rep:
            rep+=1
    circuit.measure(qr[0], cr[0])
    circuit.measure(qr[1], cr[1])
    circuit.measure(qr[2], cr[2])
    simulator = Aer.get_backend('aer_simulator')
    newcicuit = transpile(circuit, backend=simulator)
    result = simulator.run(newcicuit, shots=1000).result()
    counts = result.get_counts()
    print(counts)
    most_frequent_value = max(counts, key=counts.get)
    lastbin=most_frequent_value.zfill(5)
    notes.append(int(most_frequent_value, 2))
print(notes)







# value_counts = {}
# value_probabilities = {}
# for i in range(n):
#     statevector = result.data(0)['state_' + str(i)]
#     probabilities = Statevector(statevector)
#     # print(f"Probabilities after iteration {i}:", probabilities.probabilities_dict())
#
#     values = list(probabilities.probabilities_dict().keys())
#     weights = list(probabilities.probabilities_dict().values())
#     selected_value = np.random.choice(values, 1, p=weights)
#     selected_value = selected_value[0][-3:]
#     selected_value = int(selected_value, 2)
#     notes.append(selected_value)
#
#     for value in probabilities.probabilities_dict():
#         value = int(value[-3:], 2)
#         if value in value_counts:
#             value_counts[value] += 1
#         else:
#             value_counts[value] = 1
#     for value, weights in probabilities.probabilities_dict().items():
#         value = int(value[-3:], 2)
#         if value not in value_probabilities:
#             value_probabilities[value] = weights
#         else:
#             value_probabilities[value] += weights
# counts = result.get_counts()
# # plot_histogram(counts)
# # circuit.draw(output='mpl')
# print("Value counts:", value_counts)
# print("len value counts:", len(value_counts))
# value_probabilities = {key: val / n * 100 for key, val in value_probabilities.items()}
# value_probabilities = dict(sorted(value_probabilities.items(), key=lambda item: item[1], reverse=True))
# print("zum probabilité:", sum(value_probabilities.values()))
# print("Value probabilité:", value_probabilities)
# print("len value probabilité:", len(value_probabilities))
# print("Notes:", notes)
# sample_rate = 44100
# sequence = np.concatenate(
#     [generate_note(map_note_to_frequency(note), duration=0.5, sample_rate=sample_rate) for note in notes])
# sd.play(sequence, sample_rate)
# sd.wait()
# write("musique_generée.wav", sample_rate, sequence.astype(np.float32))
# print("Musique sauvegardée sous le nom 'musique_generée.wav'")
# circuit.draw(output='mpl').savefig("circuit.png")
# plot_value_probabilities(value_probabilities)
