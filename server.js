const express = require('express');
const path = require('path');
const axios = require('axios');

const app = express();
const port = 3000;

// API 키 설정
const apiKey = 'ㄴㄴㄴ';

// 정적 파일 제공
app.use(express.static(path.join()));

app.get('/index', (req, res) => {
    res.sendFile(path.join('index.html'));
});

app.get('/openai', async (req, res) => {
    try {
        console.log("OpenAI API 호출 시작");

        // OpenAI API에 POST 요청을 보냄
        const response = await axios.post('https://api.openai.com/v1/chat/completions', {
            model: "gpt-3.5-turbo",
            messages: [{ role: "user", content: "오늘 저녁 메뉴 추천해줘" }],
            max_tokens: 50,
        }, {
            headers: {
                'Authorization': `Bearer ${apiKey}`, //api 인증부분
                'Content-Type': 'application/json'
            }
        });
        console.log("OpenAI API 응답:", response.data.choices[0].message.content);


        // 클라이언트에 JSON 형식으로 응답을 반환
        res.json({ message: response.data.choices[0].message.content });
    } catch (error) {
        console.error("Error calling OpenAI API:", error);
        // 오류 발생 시 오류 메시지를 콘솔에 출력하고, 클라이언트에 500 상태 코드로 응답
        res.status(500).send("Error calling OpenAI API");
    }
});

app.listen(port, () => {
    console.log(`Server is running on port ${port}`);
});


