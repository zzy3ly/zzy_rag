const API = 'http://127.0.0.1:8000';
const USER_ID = 'demo-user';
const COLLECTION_ID = 'demo-petroleum';
const CONVERSATION_ID = 'demo-conversation';

const systemStatus = document.getElementById('systemStatus');
const ingestionStatus = document.getElementById('ingestionStatus');
const documentList = document.getElementById('documentList');
const messages = document.getElementById('messages');
const userInput = document.getElementById('userInput');

async function api(path, options = {}) {
  const response = await fetch(`${API}${path}`, options);
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `HTTP ${response.status}`);
  }
  return response;
}

async function postJson(path, body) {
  const response = await api(path, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body),
  });
  return response.json();
}

async function initializeDemo() {
  try {
    await postJson('/add_user/', {user_id: USER_ID});
    await postJson('/add_collection/', {
      user_id: USER_ID,
      name: '油气生产规程演示库',
      collection_id: COLLECTION_ID,
    });
    const exists = await (await api(
      `/get_conversation_exists/?conversation_id=${CONVERSATION_ID}&user_id=${USER_ID}`
    )).json();
    if (!exists.message) {
      await postJson('/add_conversation/', {
        collection_id: COLLECTION_ID,
        user_id: USER_ID,
        conversation_id: CONVERSATION_ID,
      });
    }
    systemStatus.textContent = '系统在线';
    systemStatus.className = 'badge ready';
    await refreshDocuments();
  } catch (error) {
    systemStatus.textContent = '服务连接失败';
    systemStatus.className = 'badge error';
    ingestionStatus.textContent = `初始化失败：${error.message}`;
  }
}

async function refreshDocuments() {
  const data = await (await api(
    `/get_documents_by_collection/?collection_id=${COLLECTION_ID}`
  )).json();
  documentList.replaceChildren();
  for (const doc of data.documents || []) {
    const item = documentList.appendChild(document.createElement('li'));
    item.textContent = doc.title || doc.filename || doc.document_id;
  }
  if (!documentList.children.length) {
    const item = documentList.appendChild(document.createElement('li'));
    item.textContent = '当前没有文档，请先上传。';
    item.className = 'muted';
  }
}

async function pollJob(jobId) {
  while (true) {
    const job = await (await api(`/ingestion_jobs/${jobId}`)).json();
    ingestionStatus.textContent = `入库状态：${job.status}`;
    if (job.status === 'ACTIVE') {
      ingestionStatus.textContent = `入库完成：${job.filename}`;
      await refreshDocuments();
      return;
    }
    if (['FAILED', 'CANCELLED'].includes(job.status)) {
      throw new Error(job.error || `入库任务 ${job.status}`);
    }
    await new Promise(resolve => setTimeout(resolve, 1200));
  }
}

document.getElementById('uploadButton').addEventListener('click', async () => {
  const file = document.getElementById('fileInput').files[0];
  if (!file) {
    ingestionStatus.textContent = '请先选择文件。';
    return;
  }
  const form = new FormData();
  form.append('file', file);
  form.append('collection_id', COLLECTION_ID);
  try {
    ingestionStatus.textContent = '正在提交入库任务…';
    const response = await api('/process_file/', {method: 'POST', body: form});
    const result = await response.json();
    await pollJob(result.job_id);
  } catch (error) {
    ingestionStatus.textContent = `入库失败：${error.message}`;
  }
});

function appendMessage(role, text = '') {
  const element = document.createElement('div');
  element.className = `message ${role}`;
  element.textContent = text;
  messages.appendChild(element);
  messages.scrollTop = messages.scrollHeight;
  return element;
}

async function askQuestion() {
  const question = userInput.value.trim();
  if (!question) return;
  appendMessage('user', question);
  userInput.value = '';
  const answer = appendMessage('assistant', '正在检索知识库…');
  document.getElementById('submitBtn').disabled = true;
  try {
    const response = await api('/chat/', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        user_id: USER_ID,
        message: question,
        conversation_id: CONVERSATION_ID,
      }),
    });
    answer.textContent = '';
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    while (true) {
      const {value, done} = await reader.read();
      if (done) break;
      answer.textContent += decoder.decode(value, {stream: true});
      messages.scrollTop = messages.scrollHeight;
    }
  } catch (error) {
    answer.textContent = `请求失败：${error.message}`;
  } finally {
    document.getElementById('submitBtn').disabled = false;
  }
}

document.getElementById('submitBtn').addEventListener('click', askQuestion);
userInput.addEventListener('keydown', event => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    askQuestion();
  }
});

initializeDemo();
