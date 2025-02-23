let audioContext = null;
let activeOscillators = []; // Tableau pour stocker chaque oscillateur avec son gainNode

/**
 * Démarre plusieurs oscillateurs en continu à partir d'un tableau de notes.
 * Le paramètre duration est conservé pour compatibilité externe mais n'est plus utilisé.
 *
 * @param {number[]} notes - Tableau de fréquences (en Hz) à jouer simultanément.
 * @param {number} duration - Anciennement utilisé pour la durée, maintenant ignoré.
 * @returns {Function} Une fonction permettant d'arrêter toutes les notes.
 */
export function playNotesSimultaneously(notes, duration) {
  if (!audioContext) {
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
  }

  // Arrêter et nettoyer les oscillateurs précédents
  activeOscillators.forEach(({ oscillator, gainNode }) => {
    oscillator.stop();
    oscillator.disconnect();
    gainNode.disconnect();
  });
  activeOscillators = [];

  // Pour chaque note, créer un oscillateur continu
  notes.forEach(note => {
    const oscillator = audioContext.createOscillator();
    const gainNode = audioContext.createGain();

    oscillator.frequency.setValueAtTime(note, audioContext.currentTime);
    gainNode.gain.setValueAtTime(0.5, audioContext.currentTime); // Volume à 50%

    oscillator.connect(gainNode);
    gainNode.connect(audioContext.destination);

    oscillator.start();

    activeOscillators.push({ oscillator, gainNode });
  });

  // Retourne une fonction pour arrêter toutes les notes
  return function stopNotes() {
    activeOscillators.forEach(({ oscillator, gainNode }) => {
      oscillator.stop();
      oscillator.disconnect();
      gainNode.disconnect();
    });
    activeOscillators = [];
  }
}

/**
 * Met à jour les fréquences de tous les oscillateurs en cours de manière fluide.
 *
 * @param {number[]} newFrequencies - Tableau des nouvelles fréquences (en Hz) pour chaque oscillateur.
 *                                    Il doit être de la même longueur que le tableau de notes initial.
 * @param {number} [rampTime=0.1] - Durée (en secondes) de la transition.
 */
export function updateNoteFrequencies(newFrequencies, rampTime = 0.1) {
  if (activeOscillators.length !== newFrequencies.length) {
    console.error("Le tableau newFrequencies doit avoir le même nombre d'éléments que le nombre d'oscillateurs actifs.");
    return;
  }

  activeOscillators.forEach((oscObj, index) => {
    oscObj.oscillator.frequency.linearRampToValueAtTime(
      newFrequencies[index],
      audioContext.currentTime + rampTime
    );
  });
}
