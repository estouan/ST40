document.addEventListener('DOMContentLoaded', () => {
    const groups = document.querySelectorAll('.group');
    let transitions = [];  // Liste vide pour stocker les transitions
    let currentTransition = { group: 0, image: 0 };  // Transition par défaut
    let currentTransitionIndex = 0;

    // Charger le fichier JSON
    fetch('static/transitions.json')  // Assurez-vous que le fichier est dans le bon répertoire
        .then(response => {
            if (!response.ok) {
                throw new Error('Erreur de réseau, fichier non trouvé');
            }
            return response.json();  // Convertir la réponse en JSON
        })
        .then(data => {
            transitions = data;  // Stocker les transitions dans la variable
            // Afficher les transitions dans la console
            // Initialiser avec la première transition
            groups[currentTransition.group].classList.add('visible');
            highlightImage(currentTransition.group, currentTransition.image);
        })
        .catch(error => {
            console.error('Erreur lors du chargement du fichier transitions.json:', error);
        });

    // Fonction pour mettre en avant une image dans un groupe
    function highlightImage(groupIndex, imageIndex) {
        const images = groups[groupIndex].querySelectorAll('.image');
        images.forEach(image => image.classList.remove('highlight'));  // Enlever la mise en avant de toutes les images
        images[imageIndex].classList.add('highlight');  // Mettre en avant l'image spécifiée
    }

    // Fonction pour changer de groupe et appliquer la transition
    function switchGroup() {
        if (transitions.length === 0) return;  // Si les transitions ne sont pas encore chargées, on ne fait rien

        // Masquer le groupe actuel
        groups[currentTransition.group].classList.remove('visible');

        // Passer à la prochaine transition
        currentTransitionIndex = (currentTransitionIndex + 1) % transitions.length;
        currentTransition = transitions[currentTransitionIndex];

        // Afficher le groupe correspondant
        groups[currentTransition.group].classList.add('visible');

        // Appliquer la transition pour le groupe et l'image
        highlightImage(currentTransition.group, currentTransition.image);
    }

    // Lancer l'animation de changement de groupe toutes les 2 secondes (2000ms)
    setInterval(switchGroup, 2000);
});
