document.addEventListener('DOMContentLoaded', function() {
    console.log("DOMContentLoaded event fired");

    // face-api.js 모델 로드
    Promise.all([
        faceapi.nets.tinyFaceDetector.loadFromUri('/models'),
        faceapi.nets.faceExpressionNet.loadFromUri('/models')
    ]);

    // 이미지 업로드 시 미리보기 표시
    document.getElementById('image-upload').addEventListener('change', function(event) {
        const file = event.target.files[0];
        const reader = new FileReader();
        reader.onload = function(e) {
            const img = document.getElementById('image-preview');
            img.src = e.target.result;
            img.style.display = 'block';
        };

        if (file) {
            reader.readAsDataURL(file);
        }
    });

    const uploadForm = document.getElementById("upload-form");

    // 분석 시작 버튼 클릭 시 이벤트 발생
    uploadForm.addEventListener("submit", async function(event) {
        event.preventDefault();

        const fileInput = document.getElementById('image-upload');
        const file = fileInput.files[0];

        if (!file) {
            alert('이미지를 업로드해주세요.');
            return;
        }

        const stepBox1 = document.querySelector(".step-box-1");
        stepBox1.classList.add("slide-out-left");

        stepBox1.addEventListener("animationend", function() {
            stepBox1.style.display = "none";
            console.log("Animation ended and stepBox1 hidden");
        }, { once: true });

        const modal = document.getElementById('myModal');
        modal.style.display = 'block';

        // loadingDots 애니메이션 시작
        const loadingDots = document.getElementById('loadingDots');
        const dots = ['.', '..', '...'];
        let index = 0;
        setInterval(() => {
            loadingDots.innerText = dots[index];
            index = (index + 1) % dots.length;
        }, 500);

        // 이미지가 로드된 후 감정 분석 
        const img = document.getElementById('image-preview');
        img.onload = async function() {
            try {
                const detections = await faceapi.detectAllFaces(img, new faceapi.TinyFaceDetectorOptions()).withFaceExpressions();
                displayResults(detections);
            } catch (error) {
                console.error("Error detecting faces:", error);
            }
        };

        const reader = new FileReader();
        reader.onload = function(e) {
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);
    });

    function displayResults(detections) {
        console.log("displayResults function called");
        const resultContainer = document.querySelector(".step-box-2 .result");
        resultContainer.innerHTML = '';

        if (detections.length === 0) {
            resultContainer.innerHTML = '얼굴을 감지하지 못했습니다.';
            console.log("No faces detected");
            return;
        }

        detections.forEach(detection => {
            const { expressions } = detection;
            const expressionEntries = Object.entries(expressions);

            expressionEntries.sort((a, b) => b[1] - a[1]);

            const emotionData = expressionEntries.map(([emotion, probability]) => {
                return `${emotion}: ${(probability * 100).toFixed(2)}%`;
            }).join('<br>');

            const totalProbability = expressionEntries.reduce((sum, [_, probability]) => sum + probability, 0);
            const accuracy = (totalProbability * 100).toFixed(2);

            //resultContainer.innerHTML += `감정 분석 결과:<br>${emotionData}<br>모델 신뢰도: ${accuracy}%<br><br>`;
        });
    }

    window.onclick = function(event) {
        var modal = document.getElementById('myModal');
        if (event.target == modal) {
            modal.style.display = 'none';
        }
    }
});
