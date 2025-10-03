const socket = io();

// Elemento donde mostramos el texto
const recognizedTextElement = document.getElementById('recognized-text');

// Evento de actualización de texto
socket.on('update_text', function(data) {
    recognizedTextElement.innerHTML = data.text + '<span class="cursor"></span>';
});

// Evento de conexión
socket.on('connect', function() {
    console.log('Conectado al servidor WebSocket');
});

// Limpiar texto al presionar el botón
document.getElementById('clear-text').addEventListener('click', () => {
    recognizedTextElement.innerHTML = '<span class="cursor"></span>';
});
