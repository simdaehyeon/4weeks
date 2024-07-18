document.addEventListener('DOMContentLoaded', async function () {
    console.log("DOMContentLoaded event fired");

    // 모델 로드
    const blazefaceModel = await blazeface.load();

    function startApp() {
        // 이미지 업로드 시 미리보기 표시
        document.getElementById('image-upload').addEventListener('change', function (event) {
            const file = event.target.files[0];
            const reader = new FileReader();
            reader.onload = function (e) {
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
                    const returnTensors = false;  // 얼굴 경계 상자 좌표 및 랜드마크를 반환
                    const predictions = await blazefaceModel.estimateFaces(img, returnTensors);

                    if (predictions.length > 0) {
                        // 얼굴이 감지된 경우 감정 분석 수행
                        const text = 'I love programming.'; // 감정 분석을 위한 예시 텍스트
                        const emotionPredictions = await fetchHuggingFaceEmotionAnalysis(text);

                        // 결과 확인 버튼 보이기
                        const seeResultButton = document.querySelector('.see-result');
                        seeResultButton.classList.remove('hidden');
                        const seeaiButton = document.querySelector('.see-ai');
                        seeaiButton.classList.remove('hidden');

                        // 결과 확인 버튼 클릭 시 결과 모달에 결과 표시
                        seeResultButton.addEventListener('click', function () {
                            displayResults(predictions, emotionPredictions);
                        });
                    }

                    clearInterval(loadingInterval);
                    modal.style.display = 'none';
                } catch (error) {
                    console.error("Error detecting faces:", error);
                }
            };

            const reader = new FileReader();
            reader.onload = function (e) {
                img.src = e.target.result;
            };
            reader.readAsDataURL(file);
        });

        async function fetchHuggingFaceEmotionAnalysis(text) {
            try {
                const response = await fetch('https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english', {
                    method: 'POST',
                    headers: {
                        'Authorization': 'Bearer dfdf',  // 여기에 유효한 API 키를 입력하세요.
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        inputs: text // 업데이트된 부분: inputs를 배열로 전달하지 않고 문자열로 전달
                    })
                });
                
                if (!response.ok) {
                    const errorMessage = await response.text();
                    throw new Error(`Error: ${response.status} - ${response.statusText}\n${errorMessage}`);
                }

                const result = await response.json();
                return result;
            } catch (error) {
                console.error("Error fetching emotion analysis:", error);
            }
        }

        // 감정 분석 결과 표시 함수
        function displayResults(facePredictions, emotionPredictions) {
            console.log("displayResults function called");
            const emotionResult = document.querySelector("#emotionResult"); // 감정 분석 결과 섹션 선택
            emotionResult.innerHTML = '';

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

            if (!facePredictions || facePredictions.length === 0) {
                emotionResult.innerHTML = '얼굴을 감지하지 못했습니다.';
                console.log("No faces detected");
                return;
            }

            facePredictions.forEach((prediction, index) => {
                const topLeft = prediction.topLeft;
                const bottomRight = prediction.bottomRight;
                const probability = prediction.probability;

                emotionResult.innerHTML += `얼굴 ${index + 1}:
                    <br>위치: (${topLeft[0].toFixed(2)}, ${topLeft[1].toFixed(2)}) - (${bottomRight[0].toFixed(2)}, ${bottomRight[1].toFixed(2)})
                    <br>감지 확률: ${(probability[0] * 100).toFixed(2)}%
                    <br><br>`;
            });

            if (emotionPredictions && emotionPredictions.length > 0) {
                emotionPredictions.forEach((prediction, index) => {
                    emotionResult.innerHTML += `감정 분석 결과 ${index + 1}:
                        <br>라벨: ${prediction.label}
                        <br>점수: ${(prediction.score * 100).toFixed(2)}%
                        <br><br>`;
                });
            } else {
                emotionResult.innerHTML += '감정 분석 결과를 가져오지 못했습니다.';
            }
        }
    }

    startApp();
});
