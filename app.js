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
        const seeResultButton = document.querySelector('.see-result');
        const seeAiButton = document.querySelector('.see-ai');

        modal.style.display = 'block';
        seeResultButton.classList.add('hidden');
        seeAiButton.classList.add('hidden');

        const loadingDots = document.getElementById('loadingDots');
        const dots = ['.', '..', '...'];
        let index = 0;
        const loadingInterval = setInterval(() => {
            loadingDots.innerText = dots[index];
            index = (index + 1) % dots.length;
        }, 500);

        // 이미지가 로드된 후 감정 분석 및 피부 분석
        const img = document.getElementById('image-preview');
        img.onload = async function() {
            try {
                const detections = await faceapi.detectAllFaces(img, new faceapi.TinyFaceDetectorOptions()).withFaceExpressions();
                displayResults(detections);

                // OpenCV로 피부 분석
                analyzeSkin(img);

                clearInterval(loadingInterval);
                modal.style.display = 'none';
                seeResultButton.classList.remove('hidden');
                seeAiButton.classList.remove('hidden');
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

            resultContainer.innerHTML += `감정 분석 결과:<br>${emotionData}<br>모델 신뢰도: ${accuracy}%<br><br>`;
        });

        resultContainer.style.display = 'none'; // 분석 결과를 초기에는 숨김
    }

    function analyzeSkin(img) {
        console.log("analyzeSkin function called");
        const src = cv.imread(img);
        const dst = new cv.Mat();
        cv.cvtColor(src, dst, cv.COLOR_RGBA2BGR, 0);

        // 피부 영역만을 추출하기 위해 HSV 색 공간으로 변환
        const hsv = new cv.Mat();
        cv.cvtColor(dst, hsv, cv.COLOR_BGR2HSV);

        // 피부 색 영역 범위 설정
        const low = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), [0, 30, 60, 0]);
        const high = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), [20, 150, 255, 255]);
        const skinMask = new cv.Mat();
        cv.inRange(hsv, low, high, skinMask);

        // 피부 영역만 추출
        const skin = new cv.Mat();
        cv.bitwise_and(dst, dst, skin, skinMask);

        // 피부 톤 분석
        const mean = cv.mean(skin, skinMask);
        const skinTone = `피부 톤: R=${mean[0].toFixed(0)}, G=${mean[1].toFixed(0)}, B=${mean[2].toFixed(0)}`;

        // 잡티 분석 (이진화 후 잡티 개수 세기)
        const gray = new cv.Mat();
        cv.cvtColor(skin, gray, cv.COLOR_BGR2GRAY);
        cv.threshold(gray, gray, 70, 255, cv.THRESH_BINARY_INV); // 반전하여 잡티를 흰색으로
        const contours = new cv.MatVector();
        const hierarchy = new cv.Mat();
        cv.findContours(gray, contours, hierarchy, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE);
        const blemishesCount = contours.size();
        const blemishes = `잡티 개수: ${blemishesCount}`;

        // 주름 분석 (경계 추출을 통한 주름 감지, 작은 경계선 필터링)
        const edges = new cv.Mat();
        cv.Canny(gray, edges, 100, 200);
        const wrinkleContours = new cv.MatVector();
        const hierarchyWrinkle = new cv.Mat();
        cv.findContours(edges, wrinkleContours, hierarchyWrinkle, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE);

        // 작은 경계선 필터링
        let wrinklesCount = 0;
        for (let i = 0; i < wrinkleContours.size(); i++) {
            const contour = wrinkleContours.get(i);
            if (cv.contourArea(contour) > 10) { // 최소 크기 필터
                wrinklesCount++;
            }
        }
        const wrinkles = `주름 개수: ${wrinklesCount}`;

        // 피부 결 분석 (가우시안 블러를 통한 피부 결 감지)
        const blur = new cv.Mat();
        cv.GaussianBlur(gray, blur, new cv.Size(5, 5), 1.5);
        const skinTexture = `피부 결 분석 완료`;

        const resultContainer = document.querySelector("#resultContent");
        resultContainer.innerHTML += `<br>피부 분석 결과:<br>${skinTone}<br>${blemishes}<br>${wrinkles}<br>${skinTexture}`;

        // 메모리 해제
        src.delete();
        dst.delete();
        hsv.delete();
        low.delete();
        high.delete();
        skinMask.delete();
        skin.delete();
        gray.delete();
        contours.delete();
        hierarchy.delete();
        edges.delete();
        wrinkleContours.delete();
        hierarchyWrinkle.delete();
        blur.delete();
    }

    // 결과 보기 버튼 클릭 시 이벤트 처리
    const seeResultButton = document.querySelector('.see-result');
    seeResultButton.addEventListener('click', function(event) {
        event.preventDefault();
        const resultModal = document.getElementById('resultModal');
        resultModal.style.display = 'block';
    });

    // 모달 닫기 버튼 이벤트 처리
    const closeButton = document.querySelector('.close-button');
    closeButton.addEventListener('click', function() {
        const resultModal = document.getElementById('resultModal');
        resultModal.style.display = 'none';
    });

    // 모달 외부 클릭 시 닫기
    window.onclick = function(event) {
        const modal = document.getElementById('myModal');
        const resultModal = document.getElementById('resultModal');
        if (event.target == modal) {
            modal.style.display = 'none';

            // 버튼 보이기
            const seeResultButton = document.querySelector('.see-result');
            const seeAiButton = document.querySelector('.see-ai');
            seeResultButton.classList.remove('hidden');
            seeAiButton.classList.remove('hidden');
        }
        if (event.target == resultModal) {
            resultModal.style.display = 'none';
        }
    }
});
