import pickle
from flask import Flask, render_template, request, jsonify, session, url_for



import live_music

import os
app = Flask(__name__)
app.secret_key = os.urandom(24)  # Clé secrète pour sécuriser les sessions
def get_circuit():
    try:
        # Charger le circuit depuis la session
        circuit_data = session.get('circuit')
        if circuit_data is None:
            return 'No circuit found.'

        circuit = pickle.loads(circuit_data)  # Désérialiser


        return circuit
    except Exception as e:
        print("circuit loading error")
    return None

def save_circuit(circuit):
    session['circuit'] = pickle.dumps(circuit)


# Cette fonction simule le circuit et retourne des phases



# Route pour l'interface utilisateur
@app.route('/add_gate', methods=['POST'])
def add_gate():
    print("uwu")
    try:
        # Récupérer le circuit depuis la session
        circuit = get_circuit()
        if circuit is None:
            return jsonify({'error': 'Circuit non trouvé'}), 400
        nb_qbit_note=session.get('nb_qbit_note')
        nb_qbit_ctrl = session.get('nb_qbit_ctrl')
        print(nb_qbit_note,"nb_qbit_note")
        print(nb_qbit_ctrl,"nb_qbit_ctrl")

        # Extraire les données de la requête
        data = request.get_json()
        gate_type = data.get('gate_type')

        target_qubit = int(data.get('target_qubit'))
        control_qubit=int(data.get('control_qubit'))
        angle = float(data.get('angle'))
        if target_qubit is None or control_qubit is None:
            if target_qubit is None:
                return jsonify({'message': 'no target'}), 400
            else:
                return jsonify({'message': 'no control'}), 400
        if gate_type=="CRZ" and not angle:
            return jsonify({'message': 'no angle for CRZ'}), 400
        if control_qubit < nb_qbit_note:
            return jsonify({'message': 'controlling on a note qbits'}), 400
        if target_qubit >= nb_qbit_note:
            return jsonify({'message': 'targeting a control qbits'}), 400


        # Ajouter la porte au circuit
        if gate_type == 'CNOT':


            circuit.cx(control_qubit, target_qubit)

        elif gate_type == 'CRZ':
            circuit.crz(angle,control_qubit, target_qubit)
        else:
            return jsonify({'message': 'Unknown gate type'}), 400
        tab_note=live_music.display_circuit_in_tkinter(circuit, nb_qbit_note, nb_qbit_ctrl)


        # Sauvegarder le circuit mis à jour dans la session
        save_circuit(circuit)
        public_url = url_for('static', filename='circuit.png', _external=True)

        # Retourner une représentation textuelle du circuit
        return jsonify({
            'message': 'Circuit créé avec succès !',
            'note': tab_note,  # Ajout du tableau à la réponse
            'url_circuit': public_url
        })
    except Exception as e:
        return jsonify({'message': str(e)}), 400
@app.route('/')
def index():
    return render_template('index.html')








@app.route('/create_circuit', methods=['POST'])
def create_circuit():
    try:
        # Log de débogage
        print("Requête reçue : ", request.data)

        data = request.get_json()
        if data is None:
            return jsonify({'error': 'Empty or misformed JSON body'}), 400

        # Vérification des clés
        if 'num_note_qubits' not in data or 'num_control_qubits' not in data:
            return jsonify({'error': 'Clés JSON manquantes'}), 400

        # Validation des types
        try:
            nbqbitnote = int(data['num_note_qubits'])
            nbqbitscontrole = int(data['num_control_qubits'])
        except ValueError:
            return jsonify({'message': 'Values must be integers'}), 400

        # Appel à la logique métier
        tab_note,circuit,nbqbitnote,nbqbitscontrole=live_music.create_circuit(nbqbitnote, nbqbitscontrole)
        public_url = url_for('static', filename='circuit.png', _external=True)
        session['circuit'] = pickle.dumps(circuit)
        session['nb_qbit_note']=nbqbitnote
        session['nb_qbit_ctrl']=nbqbitscontrole
        return jsonify({
            'message': 'Circuit successfully created !',
            'note': tab_note,  # Ajout du tableau à la réponse
            'url_circuit':public_url
        })

    except Exception as e:
        print("Erreur inattendue :", e)  # Pour le débogage
        return jsonify({'error creation': str(e)}), 400






if __name__ == '__main__':
    app.run(debug=True)