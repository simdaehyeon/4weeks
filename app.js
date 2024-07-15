function onOpenCvReady() {
    document.getElementById('status').innerHTML = 'ready to opencv.js.';
}

function onOpenCvError() {
    document.getElementById('status').innerHTML = 'Failed to load OpenCV.js.';
}

document.getElementById('upload-form').addEventListener('submit', function(event) {
    event.preventDefault();
    const fileInput = document.getElementById('image-upload');
    const file = fileInput.files[0];

    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            const img = new Image();
            img.src = e.target.result;
            img.onload = function() {
                // 이미지 로드 후 OpenCV.js로 처리
                const src = cv.imread(img); // 이미지 로드
                const dst = new cv.Mat();
                cv.cvtColor(src, dst, cv.COLOR_RGBA2GRAY); // 그레이스케일 변환
                cv.imshow('canvasOutput', dst); // 결과 캔버스에 출력
                src.delete(); // 메모리 해제
                dst.delete(); // 메모리 해제
            };
        };
        reader.readAsDataURL(file);
    } else {
        alert('Please select an image file.');
    }
});
