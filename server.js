const express = require('express');
const path = require('path');
const axios = require('axios');

const app = express();
const port = 3000;

// API 키 설정
const apiKey = 'ㄴㅇㄹ';

// 정적 파일 제공
app.use(express.static(path.join()));

app.get('/index', (req, res) => {
    res.sendFile(path.join('index.html'));
});

app.get('/openai', async (req, res) => {
    try {
        console.log("OpenAI API 호출 시작");
        const response = await axios.post('https://api.openai.com/v1/chat/completions', {
            model: "gpt-3.5-turbo",
            messages: [{ role: "user", content: "Hello, how are you?" }],
            max_tokens: 50,
        }, {
            headers: {
                'Authorization': `Bearer ${apiKey}`,
                'Content-Type': 'application/json'
            }
        });
        console.log("OpenAI API 응답:", response.data.choices[0].message.content);
        res.json({ message: response.data.choices[0].message.content });
    } catch (error) {
        console.error("Error calling OpenAI API:", error);
        res.status(500).send("Error calling OpenAI API");
    }
});

app.listen(port, () => {
    console.log(`Server is running on port ${port}`);
});
