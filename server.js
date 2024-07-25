const express = require('express');
const path = require('path');
const axios = require('axios');

const app = express();
const port = 3000;

// OpenAI API 키 설정
const openaiApiKey = 'sk-...'; // 실제 OpenAI API 키로 교체하세요
const googleApiKey = 'AIza...'; // 실제 Google API 키로 교체하세요

// 정적 파일 제공
app.use(express.static(path.join(__dirname, 'public')));
app.use(express.json());

app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

app.post('/openai', async (req, res) => {
    try {
        console.log("OpenAI API 호출 시작");

        const { resultsText } = req.body;

        if (!resultsText) {
            throw new Error('resultsText is undefined');
        }

        const userMessage = `얼굴 분석 결과는 다음과 같습니다:\n\n${resultsText}\n\n이 결과를 바탕으로 조언 부탁해줘 처음에는 "얼굴 분석 결과를 확인하고 조언을 드리겠습니다. 하나의 사진으로만 분석을 하여 정확한 판단을 내릴 수 없지만, 일반적인 조언을 드리겠습니다." 이 멘트 고정적으로 보여줘`;

        // OpenAI API에 POST 요청을 보냄
        const gptResponse = await axios.post('https://api.openai.com/v1/chat/completions', {
            model: "gpt-3.5-turbo",
            messages: [
                { role: "system", content: "먼저 감정에 대해서 설명하고 부정적인 감정은 최대한 긍정적이게 대답을 해줘 그리고 나이에 맞게 피부관련되서 얘기해주면 좋아 마지막에는 조언을 해줘" },
                { role: "user", content: userMessage }
            ],
            max_tokens: 2048, // 최대 글자수 설정
        }, {
            headers: {
                'Authorization': `Bearer ${openaiApiKey}`,
                'Content-Type': 'application/json'
            }
        });

        const gptResponseText = gptResponse.data.choices[0].message.content;
        console.log("OpenAI API 응답:", gptResponseText);

        // Google Text-to-Speech API 호출
        const ttsResponse = await axios.post(`https://texttospeech.googleapis.com/v1/text:synthesize?key=${googleApiKey}`, {
            input: { text: gptResponseText },
            voice: { languageCode: 'ko-KR', ssmlGender: 'FEMALE' },
            audioConfig: { audioEncoding: 'MP3' }
        });

        const audioContent = ttsResponse.data.audioContent;
        if (!audioContent) {
            throw new Error('Google TTS API 응답에 audioContent가 없습니다.');
        }
        console.log("Google Text-to-Speech API 응답 받음");

        // 클라이언트에 JSON 형식으로 응답을 반환
        res.json({ message: gptResponseText, audioContent });
    } catch (error) {
        console.error("Error calling OpenAI API or Google TTS API:", error.message);
        res.status(500).send("Error calling OpenAI API or Google TTS API");
    }
});

app.listen(port, () => {
    console.log(`Server is running on port ${port}`);
});
