let blazefaceModel;
let ageModel;
let genderModel;
let emotionModel;

document.addEventListener('DOMContentLoaded', async function () {
    console.log("DOMContentLoaded event fired");

    // 모델 로드
    blazefaceModel = await blazeface.load(); // 얼굴 인식 모델 로드
    ageModel = await tf.loadLayersModel('/web3_model/age_model/model.json'); // 나이 예측 모델 로드
    genderModel = await tf.loadLayersModel('/web3_model/gender_model/model.json'); // 성별 예측 모델 로드
    emotionModel = await tf.loadLayersModel('/tfjs_model/model.json'); // 감정 예측 모델 로드

    // 웹캠 버튼 클릭 시 이벤트
    document.getElementById('webcam-button').addEventListener('click', async function() {
        document.getElementById('webcam-container').style.display = 'block';
        const video = document.getElementById('webcam-video');
        const canvas = document.getElementById('webcam-overlay');
        const ctx = canvas.getContext('2d');

        // 웹캠 스트림 가져오기
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(stream => {
                // 웹캠 스트림을 비디오 요소에 설정
                video.srcObject = stream;
            })
            .catch(err => {
                console.error("웹캠 접근 오류:", err);
            });

        // 얼굴을 실시간으로 감지하여 경계선을 그리는 함수
        async function detectFaces() {
            if (!video.paused && !video.ended) {
                const predictions = await blazefaceModel.estimateFaces(video, false); // 얼굴 감지 부분
                ctx.clearRect(0, 0, canvas.width, canvas.height); // 이전 프레임 지우기
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height); // 비디오 프레임을 캔버스에 그리기

                ctx.lineWidth = 3;
                ctx.strokeStyle = 'green';

                  // 감정 라벨 정의
                  const emotions = ['화남', '경멸', '무서움', '행복', '슬픔', '놀람', '중립'];

                // 경계선 그리기 및 나이와 성별 예측
                for (const prediction of predictions) {
                    const [x, y] = prediction.topLeft;
                    const [right, bottom] = prediction.bottomRight;
                    const width = right - x;
                    const height = bottom - y;

                    ctx.beginPath();
                    ctx.rect(x, y, width, height);
                    ctx.stroke();

                    // 얼굴 영역 자르기
                    const faceCanvas = document.createElement('canvas');
                    faceCanvas.width = width;
                    faceCanvas.height = height;
                    const faceCtx = faceCanvas.getContext('2d');
                    faceCtx.drawImage(video, x, y, width, height, 0, 0, width, height);

                    const faceTensor = tf.browser.fromPixels(faceCanvas)
                        .resizeNearestNeighbor([96, 96])
                        .toFloat()
                        .div(tf.scalar(255.0))
                        .expandDims(0)
                        .mean(3)  
                        .expandDims(3);  

                    // 나이 및 성별 예측
                    const agePrediction = ageModel.predict(faceTensor);
                    const genderPrediction = genderModel.predict(faceTensor);
                    
                    // 나이 예측 (모델의 출력 값을 올바르게 해석)
                    const agePredictionArray = agePrediction.arraySync()[0];
                    const age = agePredictionArray.reduce((acc, prob, index) => acc + prob * (index * 10), 0); // 각 범주에 10을 곱함
                    const gender = genderPrediction.dataSync()[0] < 0.5 ? '남성' : '여성'; // 성별 예측

                    // 감정 예측
                    const emotionTensor = tf.browser.fromPixels(faceCanvas, 1)
                        .resizeNearestNeighbor([48, 48])
                        .toFloat()
                        .div(tf.scalar(255.0))
                        .expandDims();
                    const emotionPrediction = emotionModel.predict(emotionTensor).dataSync();
                    const maxEmotionIndex = emotionPrediction.indexOf(Math.max(...emotionPrediction));
                    const emotion = emotions[maxEmotionIndex]; // 가장 높은 확률의 감정

                    ctx.font = '18px Arial';
                    ctx.fillStyle = 'Yellow';
                    
                    ctx.fillText(`나이: ${Math.round(age)}`, right + 5, y);  
                    ctx.fillText(`성별: ${gender}`, right + 5, y + 20);  
                    ctx.fillText(`감정: ${emotion}`, right + 5, y + 40); 
                    console.log(`나이 예측: ${Math.round(age)}, 성별 예측: ${gender}, 감정 예측: ${emotion}`);
                }
            }
            requestAnimationFrame(detectFaces);
        }

        // 비디오와 캔버스 크기가 동일시 얼굴 감지 시작
        video.addEventListener('loadeddata', () => {
            console.log("비디오 데이터 로드됨");
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            detectFaces();
        });
    });

    document.getElementById('close-webcam-button').addEventListener('click', function() {
        let video = document.getElementById('webcam-video');
        let stream = video.srcObject;
        if (stream) {
            let tracks = stream.getTracks();
            tracks.forEach(track => track.stop());
            video.srcObject = null;
        }
        document.getElementById('webcam-container').style.display = 'none';
    });
});

function startApp() {
    // 이미지 업로드 시 미리보기 표시 및 캔버스에 그리기
    document.getElementById('image-upload').addEventListener('change', function (event) {
        const file = event.target.files[0];
        const reader = new FileReader();
        reader.onload = function (e) {
            const img = document.getElementById('image-preview');
            img.src = e.target.result;
            img.style.display = 'block';
            drawImageOnCanvas(e.target.result);
        };

        if (file) {
            reader.readAsDataURL(file);
        }
    });

    const uploadForm = document.getElementById("upload-form");

    // 분석 시작 버튼 클릭 시 이벤트 발생
    uploadForm.addEventListener("submit", async function (event) {
        event.preventDefault();

        const fileInput = document.getElementById('image-upload');
        const file = fileInput.files[0];

        if (!file) {
            alert('이미지를 업로드해주세요.');
            return;
        }

        const stepBox1 = document.querySelector(".step-box-1");
        stepBox1.classList.add("slide-out-left");

        stepBox1.addEventListener("animationend", function () {
            stepBox1.style.display = "none";
            console.log("Animation ended and stepBox1 hidden");
        }, { once: true });

        const modal = document.getElementById('myModal');
        modal.style.display = 'block';

        const loadingDots = document.getElementById('loadingDots');
        const dots = ['.', '..', '...'];
        let index = 0;
        const loadingInterval = setInterval(() => {
            loadingDots.innerText = dots[index];
            index = (index + 1) % dots.length;
        }, 500);

        // 이미지가 로드된 후 얼굴 분석
        const img = document.getElementById('image-preview');
        img.onload = async function () {
            try {
                const returnTensors = false;   
                const predictions = await blazefaceModel.estimateFaces(img, returnTensors);

                clearInterval(loadingInterval);
                modal.style.display = 'none';
                // 필터적용(96퍼 이상일때만)
                const validPredictions = predictions.filter(prediction => prediction.probability[0] >= 0.96);

                if (validPredictions.length === 0) {
                    displayNoFaceDetected();
                } else {
                    // 감정 분석 수행
                    const emotionResults = await analyzeEmotions(validPredictions, img);

                    // 나이 및 성별 예측 수행
                    const ageGenderResults = await analyzeAgeGender(validPredictions, img);

                    // 결과 확인 버튼 보이기
                    const seeResultButton = document.querySelector('.see-result');
                    seeResultButton.classList.remove('hidden');
                    const seeaiButton = document.querySelector('.see-ai');
                    seeaiButton.classList.remove('hidden');

                    // 결과 확인 버튼 클릭 시 결과 모달에 결과 표시
                    seeResultButton.addEventListener('click', function () {
                        displayResults(validPredictions, emotionResults, ageGenderResults);
                    });
                }
            } catch (error) {
                console.error("Error detecting faces:", error);
            }
        };

        const reader = new FileReader();
        reader.onload = function (e) {
            img.src = e.target.result;
            drawImageOnCanvas(e.target.result);
        };
        reader.readAsDataURL(file);
    });

    // 얼굴을 감지하지 못했을 때 표시하는 함수
    function displayNoFaceDetected() {
        console.log("No faces detected or faces have low probability");
        const emotionResult = document.querySelector("#emotionResult");
        emotionResult.innerHTML = '얼굴을 감지하지 못했습니다.';

        const resultModal = document.getElementById('resultModal');
        resultModal.style.display = 'block';

        const closeButton = document.querySelector('.close-button');
        closeButton.addEventListener('click', function () {
            resultModal.style.display = 'none';
        });

        window.onclick = function (event) {
            if (event.target == resultModal) {
                resultModal.style.display = 'none';
            }
        }
    }

    // 감정 분석 수행 함수
    async function analyzeEmotions(predictions, img) {
        const emotionResults = [];

        for (let prediction of predictions) {
            const { topLeft, bottomRight } = prediction;
            const width = bottomRight[0] - topLeft[0];
            const height = bottomRight[1] - topLeft[1];

            // 얼굴 영역 자르기
            const faceCanvas = document.createElement('canvas');
            faceCanvas.width = width;
            faceCanvas.height = height;
            const ctx = faceCanvas.getContext('2d');
            ctx.drawImage(img, topLeft[0], topLeft[1], width, height, 0, 0, width, height);

            const faceTensor = tf.browser.fromPixels(faceCanvas, 1) // 그레이스케일로 변환
                .resizeNearestNeighbor([48, 48]) 
                .toFloat()
                .div(tf.scalar(255.0))
                .expandDims();

            // 감정 분석 예측
            const emotionPrediction = await emotionModel.predict(faceTensor).dataSync();
            emotionResults.push(emotionPrediction);
        }

        return emotionResults;
    }
    async function analyzeAgeGender(predictions, img) {
        const ageGenderResults = [];
    
        for (let prediction of predictions) {
            const { topLeft, bottomRight } = prediction;
            const width = bottomRight[0] - topLeft[0];
            const height = bottomRight[1] - topLeft[1];
    
            // 얼굴 영역 자르기
            const faceCanvas = document.createElement('canvas');
            faceCanvas.width = width;
            faceCanvas.height = height;
            const ctx = faceCanvas.getContext('2d');
            ctx.drawImage(img, topLeft[0], topLeft[1], width, height, 0, 0, width, height);
    
            const faceTensor = tf.browser.fromPixels(faceCanvas, 1) // 그레이스케일로 변환
                .resizeNearestNeighbor([96, 96]) 
                .toFloat()
                .div(tf.scalar(255.0))
                .expandDims();
    
            // 나이 및 성별 예측
            const agePrediction = ageModel.predict(faceTensor);
            const genderPrediction = genderModel.predict(faceTensor);
    
            const agePredictionArray = agePrediction.arraySync()[0]; // 확률 배열로 변환
            const age = agePredictionArray.reduce((acc, prob, index) => acc + prob * (index * 10), 0); // 나이 계산
            const gender = genderPrediction.dataSync()[0]; // 성별 예측 (0: 남성, 1: 여성)
    
            // 나이를 정수로 반올림
            const adjustedAge = Math.round(age); 
    
            ageGenderResults.push({
                age: adjustedAge,
                gender: gender
            });
        }
    
        return ageGenderResults;
    }

    // 얼굴 감지 및 감정 분석 결과 표시 함수
    function displayResults(facePredictions, emotionPredictions, ageGenderResults) {
        console.log("displayResults function called");
        const emotionResult = document.querySelector("#emotionResult");
        emotionResult.innerHTML = '';

        const skinResult = document.querySelector("#skinResult");
        skinResult.innerHTML = '';

        const resultModal = document.getElementById('resultModal');
        resultModal.style.display = 'block';

        const closeButton = document.querySelector('.close-button');
        closeButton.addEventListener('click', function () {
            resultModal.style.display = 'none';
        });

        window.onclick = function (event) {
            if (event.target == resultModal) {
                resultModal.style.display = 'none';
            }
        }

        const emotions = ['화남', '경멸', '무서움', '행복', '슬픔', '놀람', '중립'];

        facePredictions.forEach((prediction, index) => {
            const probability = prediction.probability[0];

            if (probability >= 0.96) {
                const emotionPrediction = emotionPredictions[index];
                const sortedEmotions = Array.from(emotionPrediction)
                    .map((score, i) => ({ emotion: emotions[i], score }))
                    .sort((a, b) => b.score - a.score)
                    .slice(0, 3); // 상위 3개 감정만 추출

                emotionResult.innerHTML += `얼굴 ${index + 1}:
                    <br>감지 확률: ${(probability * 100).toFixed(2)}%
                    <br>해당 이미지는 얼굴을 포함하고 있습니다.
                    <br>상위 3개 감정:
                    <ul>
                        ${sortedEmotions.map(e => `<li>${e.emotion}: ${(e.score * 100).toFixed(2)}%</li>`).join('')}
                    </ul>
                    <br><br>`;

                const ageGenderPrediction = ageGenderResults[index];
                const genderText = ageGenderPrediction.gender < 0.5 ? '남성' : '여성';

                skinResult.innerHTML += `얼굴 ${index + 1}:
                    <br>예측 나이: ${ageGenderPrediction.age}
                    <br>예측 성별: ${genderText}
                    <br><br>`;
            }
        });
    }

    // "다시 검사하기" 버튼 클릭 시 초기 상태로 복원
    document.getElementById('retryButton').addEventListener('click', function () {
        console.log("Retry button clicked");
        const resultModal = document.getElementById('resultModal');
        resultModal.style.display = 'none';

        const stepBox1 = document.querySelector(".step-box-1");
        stepBox1.style.display = 'block';
        stepBox1.classList.remove("slide-out-left");

        const imgPreview = document.getElementById('image-preview');
        imgPreview.src = '';
        imgPreview.style.display = 'none';

        const seeResultButton = document.querySelector('.see-result');
        seeResultButton.classList.add('hidden');
        const seeaiButton = document.querySelector('.see-ai');
        seeaiButton.classList.add('hidden');
    });
}

// 이미지 로드를 캔버스에 그리는 함수
function drawImageOnCanvas(imageSrc) {
    const canvas = document.getElementById('overlay');
    const ctx = canvas.getContext('2d');
    const img = new Image();
    img.onload = function () {
        canvas.width = 400; // 고정된 크기
        canvas.height = 300; // 고정된 크기

        // 이미지를 중앙에 맞추기
        const scale = Math.min(canvas.width / img.width, canvas.height / img.height);
        const x = (canvas.width / 2) - (img.width / 2) * scale;
        const y = (canvas.height / 2) - (img.height / 2) * scale;
        ctx.clearRect(0, 0, canvas.width, canvas.height); // 이전 이미지 제거
        ctx.drawImage(img, x, y, img.width * scale, img.height * scale);
    };
    img.src = imageSrc;
}

startApp();
