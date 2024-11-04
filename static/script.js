document.addEventListener('DOMContentLoaded', function () {
    const uploadForm = document.getElementById('uploadForm');
    const verifyForm = document.getElementById('verifyForm');
    const statusMessageDiv = document.getElementById('statusMessage');
    const noMatchMessage = document.getElementById('noMatchMessage');
    const matchedImage = document.getElementById('matchedImage');
    const similarityScore = document.getElementById('similarityScore');

    // Upload form submission
    uploadForm.addEventListener('submit', function (e) {
        e.preventDefault();

        const formData = new FormData(uploadForm);
        fetch('/store_image', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            // Display status message
            statusMessageDiv.style.display = 'block';
            if (data.error) {
                statusMessageDiv.textContent = `Error: ${data.error}`;
                statusMessageDiv.style.color = 'red';
            } else {
                statusMessageDiv.textContent = `Image uploaded successfully. Image ID: ${data.image_id}`;
                statusMessageDiv.style.color = 'green';
            }
        })
        .catch(err => {
            statusMessageDiv.textContent = `Error: ${err}`;
            statusMessageDiv.style.color = 'red';
            statusMessageDiv.style.display = 'block';
        });
    });

    // Verify form submission
    verifyForm.addEventListener('submit', function (e) {
        e.preventDefault();

        const formData = new FormData(verifyForm);
        fetch('/verify_image', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            // Display status message
            statusMessageDiv.style.display = 'block';
            if (data.error) {
                statusMessageDiv.textContent = `Error: ${data.error}`;
                statusMessageDiv.style.color = 'red';
            } else if (data.message === 'Match found') {
                statusMessageDiv.textContent = 'Match Found For Uploaded Image';
                statusMessageDiv.style.color = 'green'
                statusMessageDiv.style.fontWeight = 'bold'
                statusMessageDiv.style.fontSize = '30px';


                matchedImage.src = `/images/${data.matched_image}`;
                matchedImage.style.display = 'block';
                similarityScore.textContent = `Similarity: ${(data.similarity * 100).toFixed(2)}%`;
                similarityScore.style.display = 'block';

                noMatchMessage.style.display = 'none';  // Hide "No match found"
            } else {
                statusMessageDiv.textContent = 'No match found for Uploaded Image';
                statusMessageDiv.style.color = 'red'
                statusMessageDiv.style.fontWeight = 'bold'
                statusMessageDiv.style.fontSize = '30px';

                noMatchMessage.style.display = 'block';  // Show "No match found"
                matchedImage.style.display = 'red';
                similarityScore.style.display = 'none';
            }
        })
        .catch(err => {
            statusMessageDiv.textContent = `Error: ${err}`;
            statusMessageDiv.style.color = 'red';
            statusMessageDiv.style.display = 'block'
            statusMessageDiv.style.fontWeight = 'bold';
        });
    });
});
