document.addEventListener('DOMContentLoaded', async function () {
    console.log("DOMContentLoaded event fired");

    // 모델 로드
    blazefaceModel = await blazeface.load(); // 얼굴 인식 모델 로드
    ageModel = await tf.loadLayersModel('/web3_model/age_model/model.json'); // 나이 예측 모델 로드
    genderModel = await tf.loadLayersModel('/web3_model/gender_model/model.json'); // 성별 예측 모델 로드
    emotionModel = await tf.loadLayersModel('/tfjs_model/model.json'); // 감정 예측 모델 로드
    skinLesionModel = await tf.loadLayersModel('Skin-Lesion-Analyzer/final_model_kaggle_version1/model.json'); // 피부 병변 분석 모델 로드

    let analysisResults = []; 
    const video = document.getElementById('webcam-video');

    document.getElementById('webcam-button').addEventListener('click', async function() {
        document.getElementById('webcam-container').style.display = 'block';
        const canvas = document.getElementById('webcam-overlay');
        const ctx = canvas.getContext('2d', { willReadFrequently: true });

        navigator.mediaDevices.getUserMedia({ video: true })
            .then(stream => {
                video.srcObject = stream;
            })
            .catch(err => {
                console.error("웹캠 접근 오류:", err);
            });
 
        // 얼굴을 실시간으로 감지
        async function detectFaces() {
            if (!video.paused && !video.ended) {
                const predictions = await blazefaceModel.estimateFaces(video, false); 
                ctx.clearRect(0, 0, canvas.width, canvas.height); 
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                ctx.lineWidth = 3;
                ctx.strokeStyle = 'green';

           
                const emotions = ['화남', '경멸', '무서움', '행복', '슬픔', '놀람', '중립'];

                analysisResults = []; 

                // 경계선 그리기 및 나이와 성별 예측
                for (const prediction of predictions) {
                    const [x, y] = prediction.topLeft;
                    const [right, bottom] = prediction.bottomRight;
                    const width = right - x;
                    const height = bottom - y;

                    ctx.beginPath();
                    ctx.rect(x, y, width, height);
                    ctx.stroke();

                    
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

                    
                    const agePrediction = ageModel.predict(faceTensor);
                    const genderPrediction = genderModel.predict(faceTensor);

                    // 나이 예측
                    const agePredictionArray = agePrediction.arraySync()[0];
                    const age = agePredictionArray.reduce((acc, prob, index) => acc + prob * (index * 10), 0); 
                    const gender = genderPrediction.dataSync()[0] < 0.5 ? '남성' : '여성';

                    // 감정 예측
                    const emotionTensor = tf.browser.fromPixels(faceCanvas, 1)
                        .resizeNearestNeighbor([48, 48])
                        .toFloat()
                        .div(tf.scalar(255.0))
                        .expandDims();
                    const emotionPrediction = emotionModel.predict(emotionTensor).dataSync();
                    const maxEmotionIndex = emotionPrediction.indexOf(Math.max(...emotionPrediction));
                    const emotion = emotions[maxEmotionIndex]; 

                    // 피부 병변 
                    const skinLesionTensor = tf.browser.fromPixels(faceCanvas)
                        .resizeNearestNeighbor([224, 224])
                        .toFloat()
                        .div(tf.scalar(255.0))
                        .expandDims();
                    const skinLesionPrediction = skinLesionModel.predict(skinLesionTensor);
                    const skinLesionResult = await skinLesionPrediction.arraySync();

                    // 피부 병변
                    const skinLesionProbabilities = skinLesionResult[0];
                    const skinLesionInfo = `종양일 확률: ${(skinLesionProbabilities[2] * 100).toFixed(2)}%`;

                    // 여드름 감지
                    const acneDetectionResult = detectAcne(faceCanvas);

                    // 주름 감지
                    const wrinkleDetectionResult = detectWrinkles(faceCanvas);

               
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

       
            const chatMessages = document.getElementById('chat-messages');
            const loadingMessage = document.createElement('div');
            loadingMessage.classList.add('chat-message', 'bot-message');
            loadingMessage.innerText = '분석 중입니다. 잠시만 기다려주세요';
            chatMessages.appendChild(loadingMessage);
            chatMessages.scrollTop = chatMessages.scrollHeight;

          
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

                     clearInterval(loadingInterval);
                    loadingMessage.innerText = data.message;
                } else {
                    console.error('GPT API request failed');
                }
            } catch (error) {
                console.error('Error sending results to GPT API:', error);
            } finally {
                document.querySelector(".step-box-3").style.display = "block";
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

// 여드름 감지 
function detectAcne(faceCanvas) {
    const faceImg = cv.imread(faceCanvas);
    const gray = new cv.Mat();
    cv.cvtColor(faceImg, gray, cv.COLOR_RGBA2GRAY);

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
        

        if (area >= 80 && area <= 200) { 
            const rect = cv.boundingRect(cnt);
            const aspectRatio = rect.width / rect.height;
            
            if (aspectRatio > 0.9 && aspectRatio < 1.1) {
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


// 주름 감지 
function detectWrinkles(faceCanvas) {
    const faceImg = cv.imread(faceCanvas);

    const hsv = new cv.Mat();
    cv.cvtColor(faceImg, hsv, cv.COLOR_RGBA2RGB);
    cv.cvtColor(hsv, hsv, cv.COLOR_RGB2HSV);

    const lowerSkin = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), [0, 20, 70, 0]);
    const upperSkin = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), [20, 255, 255, 255]);

    // 피부 영역 마스킹
    const skinMask = new cv.Mat();
    cv.inRange(hsv, lowerSkin, upperSkin, skinMask);

    // 가우시안 블러
    const blurred = new cv.Mat();
    cv.GaussianBlur(skinMask, blurred, new cv.Size(5, 5), 1.5);

    // 캐니 엣지
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
        if (area >= 20 && area <= 300) { 
            const rect = cv.boundingRect(cnt);
            const aspectRatio = rect.width / rect.height;

            if (aspectRatio > 0.5 && aspectRatio < 2.0) {
                wrinkleCount++;

                const points = cv.matFromArray(4, 1, cv.CV_32SC2, [
                    rect.x, rect.y,
                    rect.x + rect.width, rect.y,
                    rect.x + rect.width, rect.y + rect.height,
                    rect.x, rect.y + rect.height
                ]);
                const vectorOfPoints = new cv.MatVector();
                vectorOfPoints.push_back(points);
                const color = new cv.Scalar(144, 238, 144, 255); 
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

    cv.imshow(faceCanvas, faceImg);
    faceImg.delete();

    return `주름 감지 개수: ${wrinkleCount}`;
}

function startApp() {
    let emotionResults, ageGenderResults, skinLesionResults, acneResults, wrinkleResults;

    // 이미지 업로드 시 미리보기
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

    // 분석 시작 버튼
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

                    const seeResultButton = document.querySelector('.see-result');
                    seeResultButton.classList.remove('hidden');
                    const seeaiButton = document.querySelector('.see-ai');
                    seeaiButton.classList.remove('hidden');

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
                        chatMessages.scrollTop = chatMessages.scrollHeight; 

                  
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

                                 clearInterval(loadingInterval);
                                loadingMessage.innerText = data.message;
                            } else {
                                console.error('GPT API request failed');
                            }
                        } catch (error) {
                            console.error('Error sending results to GPT API:', error);
                        } finally {
                            document.querySelector(".step-box-3").style.display = "block";
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

    // 얼굴을 감지하지 못했을 때
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

    // 감정 분석 수행 
    async function analyzeEmotions(predictions, img) {
        const emotionResults = [];

        for (let prediction of predictions) {
            const { topLeft, bottomRight } = prediction;
            const width = bottomRight[0] - topLeft[0];
            const height = bottomRight[1] - topLeft[1];


            const faceCanvas = document.createElement('canvas');
            faceCanvas.width = width;
            faceCanvas.height = height;
            const ctx = faceCanvas.getContext('2d');
            ctx.drawImage(img, topLeft[0], topLeft[1], width, height, 0, 0, width, height);

            const faceTensor = tf.browser.fromPixels(faceCanvas, 1)
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
    

            const faceCanvas = document.createElement('canvas');
            faceCanvas.width = width;
            faceCanvas.height = height;
            const ctx = faceCanvas.getContext('2d');
            ctx.drawImage(img, topLeft[0], topLeft[1], width, height, 0, 0, width, height);
    
            const faceTensor = tf.browser.fromPixels(faceCanvas, 1)
                .resizeNearestNeighbor([96, 96]) 
                .toFloat()
                .div(tf.scalar(255.0))
                .expandDims();
    
            // 나이 및 성별 예측
            const agePrediction = ageModel.predict(faceTensor);
            const genderPrediction = genderModel.predict(faceTensor);
    
            const agePredictionArray = agePrediction.arraySync()[0]; 
            const age = agePredictionArray.reduce((acc, prob, index) => acc + prob * (index * 10), 0); 
            const gender = genderPrediction.dataSync()[0]; // 성별 예측 (0: 남성, 1: 여성)
    
            const adjustedAge = Math.round(age); 
    
            ageGenderResults.push({
                age: adjustedAge,
                gender: gender
            });
        }
    
        return ageGenderResults;
    }

    // 피부 병변 분석 
    async function analyzeSkinLesions(predictions, img) {
        const skinLesionResults = [];

        for (let prediction of predictions) {
            const { topLeft, bottomRight } = prediction;
            const width = bottomRight[0] - topLeft[0];
            const height = bottomRight[1] - topLeft[1];

        
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

            // 피부 병변 
            const skinLesionPrediction = await skinLesionModel.predict(faceTensor);
            const skinLesionResult = await skinLesionPrediction.array();
            skinLesionResults.push(skinLesionResult[0]); 
        }

        return skinLesionResults;
    }

    // 얼굴 감지 및 감정 분석 결과 
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
                    .slice(0, 1); 

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

    // 메시지를 채팅창에 추가
    function addMessageToChat(sender, message) {
        const chatMessages = document.getElementById('chat-messages');
        const messageElement = document.createElement('div');
        messageElement.classList.add('chat-message');
        messageElement.classList.add(sender === 'user' ? 'user-message' : 'bot-message');
        messageElement.innerText = message;
        chatMessages.appendChild(messageElement);
        chatMessages.scrollTop = chatMessages.scrollHeight;  
    }

    // 채팅폼
    const chatForm = document.getElementById('chat-form');
    chatForm.addEventListener('submit', async function (event) {
        event.preventDefault();  
        const userInput = document.getElementById('user-input');  
        const userMessage = userInput.value.trim(); 
        if (userMessage === '') return; 
        addMessageToChat('user', userMessage); 
        userInput.value = ''; 

        // GPT API 호출
        try {
            console.log('Sending message to GPT API:', userMessage);

             const emotionResultText = document.querySelector("#emotionResult").innerText;
            const skinResultText = document.querySelector("#skinResult").innerText;
            const resultsText = `${emotionResultText}\n${skinResultText}`;

            const response = await fetch('/openai-chat', { 
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ message: userMessage, resultsText })  
            });

            if (response.ok) {
                const data = await response.json(); 
                 addMessageToChat('bot', data.message);  
            } else {
                console.error('GPT API request failed with status:', response.status);
            }
        } catch (error) {
            console.error('Error sending message to GPT API:', error);
        }
    });
    document.addEventListener('DOMContentLoaded', function () {
        const retryButtonResultModal = document.getElementById('retryButton');
        if (retryButtonResultModal) {
            retryButtonResultModal.addEventListener('click', function () {
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
    
                 location.reload();
            });
        }
    
        const retryButtonChatForm = document.getElementById('retryButtonChatForm');
        if (retryButtonChatForm) {
            retryButtonChatForm.addEventListener('click', function () {
                console.log("Retry button clicked from chat form");
    
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
    
                 location.reload();
            });
        }
    });
}


// 이미지 로드를 캔버스 
function drawImageOnCanvas(imageSrc) {
    const canvas = document.getElementById('overlay');
    const ctx = canvas.getContext('2d');
    const img = new Image();
    img.onload = function () {
        canvas.width = 400;  
        canvas.height = 300; 
 
        const scale = Math.min(canvas.width / img.width, canvas.height / img.height);
        const x = (canvas.width / 2) - (img.width / 2) * scale;
        const y = (canvas.height / 2) - (img.height / 2) * scale;
        ctx.clearRect(0, 0, canvas.width, canvas.height);  
        ctx.drawImage(img, x, y, img.width * scale, img.height * scale);
    };
    img.src = imageSrc;
}

startApp();
