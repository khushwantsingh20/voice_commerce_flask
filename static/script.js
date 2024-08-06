let recordButton = document.getElementById('recordButton');
let stopButton = document.getElementById('stopButton');
let status = document.getElementById('status');
let result = document.getElementById('result');

let mediaRecorder;
let audioChunks = [];

recordButton.addEventListener('click', async () => {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });

            mediaRecorder.onstart = () => {
                audioChunks = [];
                status.textContent = 'Recording...';
                recordButton.disabled = true;
                stopButton.disabled = false;
            };

            mediaRecorder.ondataavailable = (event) => {
                audioChunks.push(event.data);
            };

            mediaRecorder.onstop = async () => {
                status.textContent = 'Recording stopped.';
                recordButton.disabled = false;
                stopButton.disabled = true;
                
                const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                const formData = new FormData();
                formData.append('audio', audioBlob, 'audio.webm');

                try {
                    const response = await fetch('/convert', {
                        method: 'POST',
                        body: formData
                    });
                    const data = await response.json();
                    if (response.ok) {
                        result.textContent = `Transcribed Text: ${data.text}`;
                    } else {
                        result.textContent = `Error: ${data.error}`;
                    }
                } catch (error) {
                    result.textContent = `Error: ${error.message}`;
                }
            };

            mediaRecorder.start();
        } catch (error) {
            status.textContent = `Error: ${error.message}`;
        }
    } else {
        status.textContent = 'getUserMedia not supported on your browser!';
    }
});

stopButton.addEventListener('click', () => {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
    }
});
