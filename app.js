document.getElementById('upload-form').addEventListener('submit', function (event) {
    event.preventDefault();
    const fileInput = document.getElementById('image-upload');
    const file = fileInput.files[0];

    if (file) {
        const reader = new FileReader();
        reader.onload = function (e) {
            const img = new Image();
            img.src = e.target.result;
            document.getElementById('result').innerHTML = '';
            document.getElementById('result').appendChild(img);

            // 여기서 이미지 처리 작업을 시작할 수 있습니다.
            console.log('Image uploaded and displayed.');
        };
        reader.readAsDataURL(file);
    } else {
        alert('Please select an image file.');
    }
});
