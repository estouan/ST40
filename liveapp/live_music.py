import qiskit
from qiskit_aer import *
from qiskit import transpile
import numpy as np
from qiskit.quantum_info import Statevector
import pygame
from scipy.io.wavfile import write

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


def run_circuit_and_measure(musicmaker):
    backend = Aer.get_backend('statevector_simulator')
    new_circuit = transpile(musicmaker, backend)  # Transpiler le circuit pour le backend choisi
    result = backend.run(new_circuit).result()
    statevector = result.get_statevector(new_circuit)  # Obtenir le vecteur d'état
    return statevector


# Fonction pour afficher le circuit dans Tkinter
def display_circuit_in_tkinter(musicmaker,nbqbitnote, nbqbitscontrole):
    statevector = run_circuit_and_measure(musicmaker)  # Obtenir le vecteur d'état
    print(statevector)
    val = choice_circuit(nbqbitscontrole)
    # val="1"
    print(nbqbitscontrole,"nbqbitscontrole")
    tab_teta = calcul_phase(statevector, val,nbqbitnote= nbqbitnote,nbqbitscontrole=nbqbitscontrole)
    tab_freq = value_to_winsound_note(tab_teta)


    # Trier les probabilités dans l'ordre décroissant

    # Redessiner le circuit
    fig = musicmaker.draw(output='mpl').savefig("static/circuit.png")
    return tab_freq



# Fonction pour ajouter une porte CNOT



# Fonction pour ajouter une porte P avec contrôle et angle en multiples de pi



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


def choice_circuit(nbqbitscontrole):
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


def calcul_phase(statevector: qiskit.quantum_info.Statevector, val, nbqbitnote, nbqbitscontrole):
    val = int(val, 2)
    selection_value = []
    for i in range(2 ** nbqbitnote):
        s=val * (2 ** nbqbitnote) + i
        print(s)
        selection_value.append(statevector[s])
    # print(selection_value,"selescction_value")
    phase_tab = []

    for i in range(2 ** nbqbitnote):
        phase_tab.append(np.angle(selection_value[i]))
    # y = 1 / np.sqrt(2 ** (nbqbitscontrole + nbqbitnote))
    y=selection_value[0]
    tab_teta = []
    for i in range(nbqbitnote):
        tab_teta.append(np.log(selection_value[2 ** i] / y) * -1j)
    return tab_teta


def create_circuit(nbqbitnote, nbqbitscontrole):
    pygame.mixer.init()
    qr = qiskit.QuantumRegister(nbqbitnote + nbqbitscontrole)
    cr = qiskit.ClassicalRegister(5)
    musicmaker = qiskit.QuantumCircuit(qr, cr)

    for i in range(nbqbitnote):
        musicmaker.h(qr[i])
        musicmaker.rx(np.pi / 2, i)
    for i in range(nbqbitscontrole):
        musicmaker.h(qr[i + nbqbitnote])
    # Lancer la boucle principale Tkinter
    tab_note=display_circuit_in_tkinter(musicmaker,nbqbitscontrole=nbqbitscontrole,nbqbitnote=nbqbitnote)
    return tab_note,musicmaker,nbqbitnote,nbqbitscontrole

