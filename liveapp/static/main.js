import {playNotesSimultaneously} from './music_function.js';

function addGate(gateName, target, control, angle) {
    $.ajax({
        url: '/add_gate',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({
            gate_type: gateName,
            target_qubit: target,
            control_qubit: control,
            angle: angle
        }),
        success: function (data) {
            if (data.error) {
                $('#message').text(data.error);  // Afficher l'erreur renvoyée par le serveur
            } else {
                $('#circuit-builder').show();
                if (data.message) {
                    // Afficher le message dans un élément HTML
                    $('#message').text(data.message);
                }
                if (data.note) {
                    // Afficher les angles dans un autre élément HTML
                    $('#note').text('Note joué: ' + data.note.join(', '));
                    playNotesSimultaneously(data.note, 0.5);
                }
                if (data.url_circuit) {
                    // Ajouter un cache-buster avec un timestamp à l'URL
                    const url = data.url_circuit + '?timestamp=' + new Date().getTime();
                    // Mettre à jour l'attribut 'src' de l'image
                    $('#circuit').attr('src', url);
                }
            }
        },
        error: function (xhr, status, error) {
            // Tenter de parser la réponse du serveur (qui semble être au format JSON)
            try {
                const response = JSON.parse(xhr.responseText);
                if (response.message) {
                    // Afficher le message d'erreur dans #message
                    $('#message').text(response.message);
                } else {
                    // Afficher un message générique si la réponse ne contient pas de message
                    $('#message').text('Erreur inconnue : ' + xhr.statusText);
                }
            } catch (e) {
                // Si la réponse ne peut pas être parsée comme JSON, afficher l'erreur brute
                $('#message').text('Erreur: ' + xhr.status + ' - ' + xhr.statusText);
            }
        }
    });
}


$(document).ready(function () {
    $('#add_cnot').on('click', function (e) {
        e.preventDefault(); // Empêcher l'envoi du formulaire ou le rechargement de la page
        const target = parseInt($('#target-qubit').val());
        const control = parseInt($('#control-qubit').val());

        if (!isNaN(target) && !isNaN(control)) {
            addGate("CNOT", target, control, 0);
        } else {

            $('#message').text('Veuillez entrer des valeurs valides pour les qubits.');
        }
    });
});

// Gestionnaire pour le bouton CRZ
$(document).ready(function () {
    $('#add_phase_gate').on('click', function (e) {
        e.preventDefault(); // Empêcher l'envoi du formulaire ou le rechargement de la page
        const target = parseInt($('#target-qubit').val());
        const control = parseInt($('#control-qubit').val());
        const angle = parseFloat($('#crz-angle').val());

        if (!isNaN(target) && !isNaN(control) && !isNaN(angle)) {
            addGate("CRZ", target, control, angle);
        } else {
            $('#message').text('Veuillez entrer des valeurs valides pour les qubits et l\'angle.');
        }

    });
});

// document.getElementById('add_cnot').addEventListener('click', addcnot);
// document.getElementById('add_phase_gate').addEventListener('click', addcrz);


$(document).ready(function () {
    $('#config-form').on('submit', function (e) {
        e.preventDefault();

        let numControlQubits = $('#num-control-qubits').val();
        let numNoteQubits = $('#num-note-qubits').val();

        $.ajax({
            url: '/create_circuit',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({
                num_control_qubits: numControlQubits,
                num_note_qubits: numNoteQubits
            }),
            success: function (data) {
                if (data.error) {
                    $('#message').text('Erreur: ' + xhr.status + ' - ' + xhr.statusText + '. Réponse: ' + xhr.responseText);
                } else {
                    $('#circuit-builder').show();
                    if (data.note) {
                        // Afficher les angles dans un autre élément HTML
                        $('#note').text('Note joué: ' + data.note.join(', '));
                        playNotesSimultaneously(data.note, 0.5);
                    }
                    if (data.message) {
                        // Afficher le message dans un élément HTML
                        $('#message').text(data.message);
                    }
                    if (data.url_circuit) {
                        // Ajouter un cache-buster avec un timestamp à l'URL
                        const url = data.url_circuit + '?timestamp=' + new Date().getTime();
                        // Mettre à jour l'attribut 'src' de l'image
                        $('#circuit').attr('src', url);
                    }
                    $('#toggle-button').click();
                }
            },
            error: function (xhr, status, error) {
                // Essayer de récupérer le message d'erreur en format JSON
                try {
                    const response = JSON.parse(xhr.responseText);
                    if (response.message) {
                        // Afficher le message d'erreur dans #message
                        $('#message').text(response.message);
                    } else {
                        // Afficher un message générique si la réponse ne contient pas de message
                        $('#message').text('Erreur inconnue : ' + xhr.statusText);
                    }
                } catch (e) {
                    // Si la réponse ne peut pas être parsée comme JSON, afficher l'erreur brute
                    $('#message').text('Erreur: ' + xhr.status + ' - ' + xhr.statusText);
                }
            }
        });
    });
});

document.addEventListener("DOMContentLoaded", () => {
    const configForm = document.getElementById("qubit-config");
    const toggleButton = document.getElementById("toggle-button");

    toggleButton.addEventListener("click", () => {
        configForm.classList.toggle("collapsed");
    });
});
