// Scroll to section
function scrollToSection(sectionId) {
    document.getElementById(sectionId).scrollIntoView({ behavior: 'smooth' });
}

// Preview image upload
function previewImage() {
    const imageInput = document.getElementById('imageFileInput');
    const imagePreview = document.getElementById('imagePreview');

    const file = imageInput.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            imagePreview.src = e.target.result;
            imagePreview.style.display = 'block';
        };
        reader.readAsDataURL(file);
    } else {
        imagePreview.style.display = 'none';
    }
}

// Submit audio file (placeholder function for integration with Flask)
function submitAudio() {
    const audioFileInput = document.getElementById('audioFileInput');
    if (!audioFileInput.files.length) {
        alert("Please upload an audio file before submitting.");
        return;
    }
    // Code to submit audio file (to be integrated with Flask backend)
    alert("Audio file submitted for prediction.");
}

// Submit image file (placeholder function for integration with Flask)
function submitImage() {
    const imageFileInput = document.getElementById('imageFileInput');
    if (!imageFileInput.files.length) {
        alert("Please upload an image file before submitting.");
        return;
    }
    // Code to submit image file (to be integrated with Flask backend)
    alert("Image file submitted for prediction.");
}
