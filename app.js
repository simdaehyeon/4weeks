document.addEventListener('DOMContentLoaded', async function () {
    console.log("DOMContentLoaded event fired");

    // 모델 로드
    blazefaceModel = await blazeface.load(); // 얼굴 인식 모델 로드
    ageModel = await tf.loadLayersModel('/web3_model/age_model/model.json'); // 나이 예측 모델 로드
    genderModel = await tf.loadLayersModel('/web3_model/gender_model/model.json'); // 성별 예측 모델 로드
    emotionModel = await tf.loadLayersModel('/tfjs_model/model.json'); // 감정 예측 모델 로드
    skinLesionModel = await tf.loadLayersModel('Skin-Lesion-Analyzer/final_model_kaggle_version1/model.json'); // 피부 병변 분석 모델 로드

    let analysisResults = []; // 분석 결과 저장
    const video = document.getElementById('webcam-video');

    // 웹캠 버튼 클릭 시 이벤트
    document.getElementById('webcam-button').addEventListener('click', async function() {
        document.getElementById('webcam-container').style.display = 'block';
        const canvas = document.getElementById('webcam-overlay');
        const ctx = canvas.getContext('2d', { willReadFrequently: true });

        // 웹캠 스트림 가져오기
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(stream => {
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

                analysisResults = []; // 초기화

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
                    const faceCtx = faceCanvas.getContext('2d', { willReadFrequently: true });
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

                    // 피부 병변 예측
                    const skinLesionTensor = tf.browser.fromPixels(faceCanvas)
                        .resizeNearestNeighbor([224, 224])
                        .toFloat()
                        .div(tf.scalar(255.0))
                        .expandDims();
                    const skinLesionPrediction = skinLesionModel.predict(skinLesionTensor);
                    const skinLesionResult = await skinLesionPrediction.arraySync();

                    // 피부 병변 확률 값 추출
                    const skinLesionProbabilities = skinLesionResult[0];
                    const skinLesionInfo = `종양일 확률: ${(skinLesionProbabilities[2] * 100).toFixed(2)}%`;

                    // 여드름 감지
                    const acneDetectionResult = detectAcne(faceCanvas);

                    // 주름 감지
                    const wrinkleDetectionResult = detectWrinkles(faceCanvas);

                    // 결과 저장
                    analysisResults.push({
                        age: Math.round(age),
                        gender: gender,
                        emotion: emotion,
                        skinLesionInfo: skinLesionInfo,
                        acne: acneDetectionResult,
                        wrinkles: wrinkleDetectionResult
                    });

                    ctx.font = '18px Arial';
                    ctx.fillStyle = 'Yellow';

                    ctx.fillText(`나이: ${Math.round(age)}`, right + 5, y);
                    ctx.fillText(`성별: ${gender}`, right + 5, y + 20);
                    ctx.fillText(`감정: ${emotion}`, right + 5, y + 40);
                    ctx.fillText(skinLesionInfo, right + 5, y + 60);
                    ctx.fillText(acneDetectionResult, right + 5, y + 80);
                    ctx.fillText(wrinkleDetectionResult, right + 5, y + 100);

                    console.log(`나이 예측: ${Math.round(age)}, 성별 예측: ${gender}, 감정 예측: ${emotion}, ${skinLesionInfo}, ${acneDetectionResult}, ${wrinkleDetectionResult}`);
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

        // "분석하기" 버튼 클릭 이벤트
        document.getElementById('gpt-analyze').addEventListener('click', async function() {
            const stepBox1 = document.querySelector(".step-box-1");
            stepBox1.style.display = "none";
             const stepBox2 = document.querySelector(".step-box-2");
            stepBox2.style.display = "none";
 
            // 웹캠 스트림 정지
            let stream = video.srcObject;
            if (stream) {
                let tracks = stream.getTracks();
                tracks.forEach(track => track.stop());
                video.srcObject = null;
            }
            document.getElementById('webcam-container').style.display = 'none';

            // "분석 중입니다" 메시지 추가
            const chatMessages = document.getElementById('chat-messages');
            const loadingMessage = document.createElement('div');
            loadingMessage.classList.add('chat-message', 'bot-message');
            loadingMessage.innerText = '분석 중입니다. 잠시만 기다려주세요';
            chatMessages.appendChild(loadingMessage);
            chatMessages.scrollTop = chatMessages.scrollHeight; // 스크롤을 맨 아래로 이동

            // "분석 중입니다" 애니메이션 추가
            const loadingDots = document.createElement('span');
            loadingMessage.appendChild(loadingDots);
            const dots = ['.', '..', '...'];
            let index = 0;
            const loadingInterval = setInterval(() => {
                loadingDots.innerText = dots[index];
                index = (index + 1) % dots.length;
            }, 500);

            const resultsText = analysisResults.map(result => {
                return `나이: ${result.age}, 성별: ${result.gender}, 감정: ${result.emotion}, ${result.skinLesionInfo}, ${result.acne}, ${result.wrinkles}`;
            }).join('\n');
            
            try {
                const response = await fetch('/openai', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ resultsText })
                });

                if (response.ok) {
                    const data = await response.json();
                    console.log("Received response from GPT API:", data.message);

                    // "분석 중입니다" 메시지를 결과로 교체
                    clearInterval(loadingInterval);
                    loadingMessage.innerText = data.message;
                } else {
                    console.error('GPT API request failed');
                }
            } catch (error) {
                console.error('Error sending results to GPT API:', error);
            } finally {
                document.querySelector(".step-box-3").style.display = "block"; // step-box-3 표시
            }
        });
    });

    document.getElementById('close-webcam-button').addEventListener('click', function() {
        let stream = video.srcObject;
        if (stream) {
            let tracks = stream.getTracks();
            tracks.forEach(track => track.stop());
            video.srcObject = null;
        }
        document.getElementById('webcam-container').style.display = 'none';
    });
});

// 여드름 감지 함수
function detectAcne(faceCanvas) {
    const faceImg = cv.imread(faceCanvas);
    const gray = new cv.Mat();
    cv.cvtColor(faceImg, gray, cv.COLOR_RGBA2GRAY);

    // 이미지 해상도 조정
    const highRes = new cv.Mat();
    cv.resize(gray, highRes, new cv.Size(640, 640));

    const th3 = new cv.Mat();
    cv.adaptiveThreshold(highRes, th3, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 15, 5);
    const dilated = new cv.Mat();
    const kernel = cv.Mat.ones(1, 1, cv.CV_8UC1);
    cv.dilate(th3, dilated, kernel, new cv.Point(-1, -1), 2);

    const contours = new cv.MatVector();
    const hierarchy = new cv.Mat();
    cv.findContours(dilated, contours, hierarchy, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE);

    let acneCount = 0;
    for (let i = 0; i < contours.size(); ++i) {
        const cnt = contours.get(i);
        const area = cv.contourArea(cnt);
        
        // 여드름 크기의 범위 및 모양 기준 조정
        if (area >= 50 && area <= 300) { // 더 엄격한 크기 범위
            const rect = cv.boundingRect(cnt);
            const aspectRatio = rect.width / rect.height;
            
            // 모양 기준 추가 (가로 세로 비율이 일정 범위 내에 있는지 확인)
            if (aspectRatio > 0.75 && aspectRatio < 1.25) {
                acneCount++;
            }
        }
        cnt.delete();
    }

    contours.delete();
    hierarchy.delete();
    gray.delete();
    th3.delete();
    dilated.delete();
    kernel.delete();
    faceImg.delete();
    
    return `여드름 감지 개수: ${acneCount}`;
}

function detectWrinkles(faceCanvas) {
    const faceImg = cv.imread(faceCanvas);

    // HSV 색상 공간으로 변환
    const hsv = new cv.Mat();
    cv.cvtColor(faceImg, hsv, cv.COLOR_RGBA2RGB);
    cv.cvtColor(hsv, hsv, cv.COLOR_RGB2HSV);

    // 피부 색상 범위 설정 (Hue, Saturation, Value 범위)
    const lowerSkin = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), [0, 20, 70, 0]);
    const upperSkin = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), [20, 255, 255, 255]);

    // 피부 영역 마스킹
    const skinMask = new cv.Mat();
    cv.inRange(hsv, lowerSkin, upperSkin, skinMask);

    // 가우시안 블러 적용
    const blurred = new cv.Mat();
    cv.GaussianBlur(skinMask, blurred, new cv.Size(5, 5), 1.5);

    // 캐니 엣지 감지기 사용
    const edges = new cv.Mat();
    cv.Canny(blurred, edges, 50, 150);

    const contours = new cv.MatVector();
    const hierarchy = new cv.Mat();
    cv.findContours(edges, contours, hierarchy, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE);

    let wrinkleCount = 0;
    for (let i = 0; i < contours.size(); ++i) {
        const cnt = contours.get(i);
        const area = cv.contourArea(cnt);

        // 주름 크기 및 모양 기준 조정
        if (area >= 30 && area <= 200) { // 주름의 크기 범위 설정
            const rect = cv.boundingRect(cnt);
            const aspectRatio = rect.width / rect.height;

            // 모양 기준 추가 (가로 세로 비율이 일정 범위 내에 있는지 확인)
            if (aspectRatio > 0.2 && aspectRatio < 5) {
                wrinkleCount++;
                // 주름 라인을 연한 초록색으로 그리기
                const points = cv.matFromArray(4, 1, cv.CV_32SC2, [
                    rect.x, rect.y,
                    rect.x + rect.width, rect.y,
                    rect.x + rect.width, rect.y + rect.height,
                    rect.x, rect.y + rect.height
                ]);
                const vectorOfPoints = new cv.MatVector();
                vectorOfPoints.push_back(points);
                const color = new cv.Scalar(144, 238, 144, 255); // 연한 초록색
                cv.polylines(faceImg, vectorOfPoints, true, color, 2);
                points.delete();
                vectorOfPoints.delete();
            }
        }
        cnt.delete();
    }

    contours.delete();
    hierarchy.delete();
    hsv.delete();
    blurred.delete();
    edges.delete();
    skinMask.delete();
    lowerSkin.delete();
    upperSkin.delete();

    cv.imshow(faceCanvas, faceImg); // 주름이 그려진 이미지를 캔버스에 출력
    faceImg.delete();

    return `주름 감지 개수: ${wrinkleCount}`;
}

function startApp() {
    let emotionResults, ageGenderResults, skinLesionResults, acneResults, wrinkleResults;

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
                    emotionResults = await analyzeEmotions(validPredictions, img);

                    // 나이 및 성별 예측 수행
                    ageGenderResults = await analyzeAgeGender(validPredictions, img);

                    // 피부 병변 예측 수행
                    skinLesionResults = await analyzeSkinLesions(validPredictions, img);

                    // 여드름 감지 수행
                    acneResults = validPredictions.map(prediction => {
                        const { topLeft, bottomRight } = prediction;
                        const width = bottomRight[0] - topLeft[0];
                        const height = bottomRight[1] - topLeft[1];

                        const faceCanvas = document.createElement('canvas');
                        faceCanvas.width = width;
                        faceCanvas.height = height;
                        const ctx = faceCanvas.getContext('2d');
                        ctx.drawImage(img, topLeft[0], topLeft[1], width, height, 0, 0, width, height);

                        return detectAcne(faceCanvas);
                    });

                    // 주름 감지 수행
                    wrinkleResults = validPredictions.map(prediction => {
                        const { topLeft, bottomRight } = prediction;
                        const width = bottomRight[0] - topLeft[0];
                        const height = bottomRight[1] - topLeft[1];

                        const faceCanvas = document.createElement('canvas');
                        faceCanvas.width = width;
                        faceCanvas.height = height;
                        const ctx = faceCanvas.getContext('2d');
                        ctx.drawImage(img, topLeft[0], topLeft[1], width, height, 0, 0, width, height);

                        return detectWrinkles(faceCanvas);
                    });

                    // 결과 확인 버튼 보이기
                    const seeResultButton = document.querySelector('.see-result');
                    seeResultButton.classList.remove('hidden');
                    const seeaiButton = document.querySelector('.see-ai');
                    seeaiButton.classList.remove('hidden');

                    // 결과 확인 버튼 클릭 시 결과 모달에 결과 표시
                    seeResultButton.addEventListener('click', function () {
                        displayResults(validPredictions, emotionResults, ageGenderResults, skinLesionResults, acneResults, wrinkleResults);
                    });

                    // GPT API를 통해 분석 결과 전송
                    seeaiButton.addEventListener('click', async function () {
                        const stepBox2 = document.querySelector('.step-box-2');
                        stepBox2.classList.add('slide-out-left');

                        stepBox2.addEventListener('animationend', function() {
                            stepBox2.style.display = 'none';
                            console.log("step-box-2 hidden");
                        }, { once: true });

                        const emotionResultText = document.querySelector("#emotionResult").innerText;
                        const skinResultText = document.querySelector("#skinResult").innerText;
                        const resultsText = `${emotionResultText}\n${skinResultText}`;

                        console.log("Sending results to GPT API:", resultsText);

                        // "분석 중입니다" 메시지 추가
                        const chatMessages = document.getElementById('chat-messages');
                        const loadingMessage = document.createElement('div');
                        loadingMessage.classList.add('chat-message', 'bot-message');
                        loadingMessage.innerText = '분석 중입니다. 잠시만 기다려주세요';
                        chatMessages.appendChild(loadingMessage);
                        chatMessages.scrollTop = chatMessages.scrollHeight; // 스크롤을 맨 아래로 이동

                        // "분석 중입니다" 애니메이션 추가
                        const loadingDots = document.createElement('span');
                        loadingMessage.appendChild(loadingDots);
                        const dots = ['.', '..', '...'];
                        let index = 0;
                        const loadingInterval = setInterval(() => {
                            loadingDots.innerText = dots[index];
                            index = (index + 1) % dots.length;
                        }, 500);

                        try {
                            const response = await fetch('/openai', {
                                method: 'POST',
                                headers: {
                                    'Content-Type': 'application/json'
                                },
                                body: JSON.stringify({ resultsText })
                            });

                            if (response.ok) {
                                const data = await response.json();
                                console.log("Received response from GPT API:", data.message);

                                // "분석 중입니다" 메시지를 결과로 교체
                                clearInterval(loadingInterval);
                                loadingMessage.innerText = data.message;
                            } else {
                                console.error('GPT API request failed');
                            }
                        } catch (error) {
                            console.error('Error sending results to GPT API:', error);
                        } finally {
                            document.querySelector(".step-box-3").style.display = "block"; // step-box-3 표시
                        }
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

    // 피부 병변 분석 수행 함수
    async function analyzeSkinLesions(predictions, img) {
        const skinLesionResults = [];

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

            const faceTensor = tf.browser.fromPixels(faceCanvas)
                .resizeNearestNeighbor([224, 224])
                .toFloat()
                .div(tf.scalar(255.0))
                .expandDims();

            // 피부 병변 예측
            const skinLesionPrediction = await skinLesionModel.predict(faceTensor);
            const skinLesionResult = await skinLesionPrediction.array();
            skinLesionResults.push(skinLesionResult[0]); // 결과 배열의 첫 번째 요소
        }

        return skinLesionResults;
    }

    // 얼굴 감지 및 감정 분석 결과 표시 함수
    function displayResults(facePredictions, emotionPredictions, ageGenderResults, skinLesionResults, acneResults, wrinkleResults) {
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
                    .slice(0, 1); // 상위 1개 감정만 추출

                emotionResult.innerHTML += `얼굴 ${index + 1}:
                    <br>감지 확률: ${(probability * 100).toFixed(2)}%
                    <br>해당 이미지는 얼굴을 포함하고 있습니다.
                    <br>상위 감정:
                    <ul>
                        ${sortedEmotions.map(e => `<li>${e.emotion}: ${(e.score * 100).toFixed(2)}%</li>`).join('')}
                    </ul>
                    <br><br>`;

                const ageGenderPrediction = ageGenderResults[index];
                const genderText = ageGenderPrediction.gender < 0.5 ? '남성' : '여성';

                // 피부 병변 결과 처리
                const skinLesionPrediction = skinLesionResults[index];
                const skinLesionInfo = `종양 의심 확률: ${(skinLesionPrediction[2] * 100).toFixed(2)}%`;

                // 여드름 감지 결과
                const acneInfo = acneResults[index];

                // 주름 감지 결과
                const wrinkleInfo = wrinkleResults[index];

                skinResult.innerHTML += `얼굴 ${index + 1}:
                    <br>나이: ${ageGenderPrediction.age}
                    <br>성별: ${genderText}
                    <br>${skinLesionInfo}
                    <br>${acneInfo}
                    <br>${wrinkleInfo}
                    <br><br>`;
            }
        });
    }

    // 메시지를 채팅창에 추가하는 함수
    function addMessageToChat(sender, message) {
        const chatMessages = document.getElementById('chat-messages');
        const messageElement = document.createElement('div');
        messageElement.classList.add('chat-message');
        messageElement.classList.add(sender === 'user' ? 'user-message' : 'bot-message');
        messageElement.innerText = message;
        chatMessages.appendChild(messageElement);
        chatMessages.scrollTop = chatMessages.scrollHeight; // 스크롤을 맨 아래로 이동
    }

    // 채팅폼 이벤트 리스너
    const chatForm = document.getElementById('chat-form');
    chatForm.addEventListener('submit', async function (event) {
        event.preventDefault(); // 폼 제출 기본 동작을 방지 (페이지 새로고침 방지)
        const userInput = document.getElementById('user-input'); // 입력란 요소를 가져옴
        const userMessage = userInput.value.trim(); // 입력된 메시지를 가져와서 앞뒤 공백 제거
        if (userMessage === '') return; // 메시지가 비어있으면 함수 종료
        addMessageToChat('user', userMessage); // 입력된 메시지를 채팅창에 사용자 메시지로 추가
        userInput.value = ''; // 입력란을 비움

        // GPT API 호출
        try {
            console.log('Sending message to GPT API:', userMessage);

            // 분석 결과와 사용자 메시지를 함께 전송
            const emotionResultText = document.querySelector("#emotionResult").innerText;
            const skinResultText = document.querySelector("#skinResult").innerText;
            const resultsText = `${emotionResultText}\n${skinResultText}`;

            const response = await fetch('/openai-chat', { // 새로운 엔드포인트로 요청 전송
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ message: userMessage, resultsText }) // 메시지와 결과를 함께 전송
            });

            if (response.ok) {
                const data = await response.json(); // 응답 데이터를 JSON 형식으로 파싱
                 addMessageToChat('bot', data.message); // GPT 응답 메시지를 채팅창에 봇 메시지로 추가
            } else {
                console.error('GPT API request failed with status:', response.status);
            }
        } catch (error) {
            console.error('Error sending message to GPT API:', error);
        }
    });

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

    // 페이지 새로고침
    location.reload();
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
