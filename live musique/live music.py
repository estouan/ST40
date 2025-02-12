import qiskit
from qiskit_aer import *
from qiskit import transpile
import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from qiskit.quantum_info import Statevector
import pygame
from scipy.io.wavfile import write

# Définir les paramètres du circuit
nbqbitscontrole = 1
nbqbitnote = 4
# variables globales
save_counter = 0
probability_labels = []
sound_objects = []


def update_probabilities(probabilities):
    print(probabilities, "probabilities")
    global probability_labels
    # Détruire les anciens labels
    for label in probability_labels:
        label.destroy()
    probability_labels = []

    # Créer de nouveaux labels pour les probabilités
    for i, (val, prob) in enumerate(probabilities.items()):
        label = tk.Label(values_frame,
                         text=f"valeur de sortie {val}: {prob * 100:.2f}%")  # Utiliser values_frame comme parent
        label.grid(row=i, column=2)
        probability_labels.append(label)


def generate_note_wav(frequency, duration=1.0, sample_rate=44100, amplitude=0.5):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = amplitude * np.sin(2 * np.pi * frequency * t)

    # Créer un fichier WAV temporaire pour cette note
    filename = f"note_{int(frequency)}.wav"
    write(filename, sample_rate, wave.astype(np.float32))
    return filename


def play_note_continuously(frequency):
    note_file = generate_note_wav(frequency, duration=5.0)  # Fichier temporaire pour 5 secondes
    sound = pygame.mixer.Sound(note_file)
    sound.play(loops=-1)  # Joue en boucle infinie
    return sound


def stop_note():
    global sound_objects
    for sound in sound_objects:
        sound.stop()
    sound_objects = []


def run_circuit_and_measure():
    backend = Aer.get_backend('statevector_simulator')
    new_circuit = transpile(musicmaker, backend)  # Transpiler le circuit pour le backend choisi
    result = backend.run(new_circuit).result()
    statevector = result.get_statevector(new_circuit)  # Obtenir le vecteur d'état
    return statevector


# Fonction pour afficher le circuit dans Tkinter
def display_circuit_in_tkinter():
    global save_counter
    save_counter += 1
    statevector = run_circuit_and_measure()  # Obtenir le vecteur d'état
    print(statevector)
    val = choice_circuit()
    # val="1"
    tab_teta = calcul_phase(statevector, val)
    tab_freq = value_to_winsound_note(tab_teta)
    play_note(tab_freq)

    # Trier les probabilités dans l'ordre décroissant

    # Redessiner le circuit
    global canvas
    fig = musicmaker.draw(output='mpl')
    if canvas:
        canvas.get_tk_widget().destroy()  # Détruire l'ancien canvas avant d'ajouter le nouveau
    canvas = FigureCanvasTkAgg(fig, master=circuit_frame)  # Utiliser circuit_frame comme parent
    canvas.draw()
    canvas.get_tk_widget().pack()


# Fonction pour ajouter une porte CNOT
def add_cnot():
    try:
        control_qubit = int(entry_control.get())
        target_qubit = int(entry_target.get())
        musicmaker.cx(qr[control_qubit], qr[target_qubit])  # Ajouter une porte CNOT
        display_circuit_in_tkinter()  # Redessiner le circuit
    except (ValueError, IndexError):
        print("Erreur : veuillez entrer des qubits valides.")


# Fonction pour ajouter une porte P avec contrôle et angle en multiples de pi
def add_p():
    try:
        control_qubit = int(entry_control.get())
        target_qubit = int(entry_target.get())
        p_value = float(entry_p_value.get()) * np.pi  # Convertir l'angle en multiples de pi
        musicmaker.cp(p_value, qr[control_qubit], qr[target_qubit])  # Ajouter une porte P avec contrôle
        display_circuit_in_tkinter()  # Redessiner le circuit
    except (ValueError, IndexError):
        print("Erreur : veuillez entrer des valeurs valides pour la porte P.")


def get_freq(freq):
    if freq == 0:
        return 261.63
    elif freq == 1:
        return 293.66
    elif freq == 2:
        return 329.63
    elif freq == 3:
        return 349.23
    elif freq == 4:
        return 392.00
    elif freq == 5:
        return 440.00
    elif freq == 6:
        return 493.88
    elif freq == 7:
        return 523.25
    else:
        return 0


def choice_circuit():
    qr_choicecircuit = qiskit.QuantumRegister(nbqbitscontrole)
    cr_choicecircuit = qiskit.ClassicalRegister(nbqbitscontrole)
    qc = qiskit.QuantumCircuit(qr_choicecircuit, cr_choicecircuit)
    # oracle
    qc.h(qr_choicecircuit)
    # mesure
    qc.measure(qr_choicecircuit, cr_choicecircuit)
    simulator = Aer.get_backend('qasm_simulator')
    newcircuit = transpile(qc, simulator)
    result = simulator.run(newcircuit, shots=1024).result()
    counts = result.get_counts()
    print(counts)
    max_counted_value = max(counts, key=counts.get)
    print(max_counted_value, "max_counted_value")
    return max_counted_value


def value_to_winsound_note(tab_teta):
    # Convertir les phases en fréquences
    tab_freq = []
    min_freq = 100  # Fréquence minimale en Hz
    max_freq = 2000  # Fréquence maximale en Hz
    base_freq = 440  # La fréquence de référence (La4)
    for index, teta in enumerate(tab_teta):
        # Convertir la phase en fréquence
        freq = teta.real % (2 * np.pi) / (np.pi / 4)
        freq = round(freq)
        freq = get_freq(freq)
        # Limiter la fréquence entre min_freq et max_freq
        freq = max(min_freq, min(max_freq, freq))
        # Ajuster la fréquence pour qu'elle corresponde à une note musicale standard
        note = round(12 * np.log2(freq / base_freq))
        freq = base_freq * 2 ** (note / 12)
        print(freq, "freq")
        tab_freq.append(freq)

    return tab_freq


def calcul_phase(statevector: qiskit.quantum_info.Statevector, val):
    # print(val,"val")
    val = int(val, 2)
    print(val, "val")
    selection_value = []
    for i in range(2 ** nbqbitnote):
        selection_value.append(statevector[val * (2 ** nbqbitnote) + i])
    # print(selection_value,"selescction_value")
    phase_tab = []
    for i in range(2 ** nbqbitnote):
        phase_tab.append(np.angle(selection_value[i]))
    # print(phase_tab,"phase_tab")
    # phase_qbit1 = np.angle(selection_value[0]+selection_value[1])
    # phase_qbit2 = np.angle(selection_value[0]+selection_value[2])
    # print(phase_qbit1,phase_qbit2,"phase_qbit1,phase_qbit2")
    print(selection_value, "selection_value")
    y = 1 / np.sqrt(2 ** (nbqbitscontrole + nbqbitnote))
    print(y, "y")
    tab_teta = []
    for i in range(nbqbitnote):
        tab_teta.append(np.log(selection_value[2 ** i] / y) * -1j)

    print(tab_teta, "tab_teta")
    return tab_teta


def play_note(tab_freq):
    global sound_objects
    stop_note()
    # Arrêter tous les sons en cours de lecture
    for freq in tab_freq:
        sound = play_note_continuously(freq)
        sound_objects.append(sound)

    # Ajouter une note à la piste
    notes_text = " et ".join([f"{freq:.2f} Hz" for freq in tab_freq])
    label_note_choisie.config(text=f"Note choisie: {notes_text}")




def on_closing():
    root.destroy()
    pygame.quit()


# Initialiser l'interface Tkinter
root = tk.Tk()
root.title("Music Maker")
root.geometry("800x600")
root.protocol("WM_DELETE_WINDOW", on_closing)
rows, cols = 5, 5
cell_size = 50
pygame.mixer.init()

# Créer un fichier MIDI


# Ajouter une piste au fichier MIDI


canvas = None  # Pour gérer le canvas

# Création du circuit quantique

qr = qiskit.QuantumRegister(nbqbitnote + nbqbitscontrole)
cr = qiskit.ClassicalRegister(5)
musicmaker = qiskit.QuantumCircuit(qr, cr)

for i in range(nbqbitnote):
    musicmaker.h(qr[i])
    musicmaker.rx(np.pi / 2, i)
for i in range(nbqbitscontrole):
    musicmaker.h(qr[i + nbqbitnote])

circuit_frame = tk.Frame(root)
circuit_frame.grid(row=0, column=0)

values_frame = tk.Frame(root)
values_frame.grid(row=0, column=1)

# Afficher le circuit initial


# Widgets for the CNOT and P gates
label_control = tk.Label(values_frame, text="Qubit de contrôle (CNOT et P):")
label_control.grid(row=0, column=0)
entry_control = tk.Entry(values_frame)
entry_control.grid(row=1, column=0)

label_target = tk.Label(values_frame, text="Qubit cible (CNOT et P):")
label_target.grid(row=2, column=0)
entry_target = tk.Entry(values_frame)
entry_target.grid(row=3, column=0)

# Button to add a CNOT gate
button_cnot = tk.Button(values_frame, text="Ajouter CNOT", command=add_cnot)
button_cnot.grid(row=4, column=0)

# Widgets for the P gate
label_p_value = tk.Label(values_frame, text="Valeur de la porte P (en multiples de π):")
label_p_value.grid(row=5, column=0)
entry_p_value = tk.Entry(values_frame)
entry_p_value.grid(row=6, column=0)

# Button to add a P gate
button_p = tk.Button(values_frame, text="Ajouter P", command=add_p)
button_p.grid(row=7, column=0)

#
label_note_choisie = tk.Label(values_frame, text="Note choisie: ")
label_note_choisie.grid(row=8, column=0)
# Lancer la boucle principale Tkinter
display_circuit_in_tkinter()
root.mainloop()
