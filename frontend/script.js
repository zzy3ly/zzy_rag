document.getElementById('submitBtn').addEventListener('click', async function() {
    const userInput = document.getElementById('userInput').value;
    const responseDiv = document.getElementById('response');

    if (!userInput) {
        responseDiv.innerHTML = '请先输入问题。';
        return;
    }

    responseDiv.innerHTML = '正在获取回答...';

    try {
        const res = await fetch('http://localhost:8000/get_collections_by_user/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ user_id: 'user1' })
        });

        if (!res.ok) {
            throw new Error('网络错误');
        }

        const data = await res.json();
        const preloadedQuestionsDiv = document.getElementById('preloadedQuestions');
        data.collections.forEach(item => {
            const questionElement = document.createElement('div');
            questionElement.innerHTML = `<strong>问题:</strong> ${item.question} <br> <strong>答案:</strong> ${item.answer}`;
            preloadedQuestionsDiv.appendChild(questionElement);
        });
    } catch (error) {
        responseDiv.innerHTML = '发生错误: ' + error.message;
    }
});
