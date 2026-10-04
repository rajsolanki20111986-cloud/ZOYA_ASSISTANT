import express from 'express';
import cors from 'cors';

const app = express();
const PORT = 3000;

app.use(cors());
app.use(express.json());

let server = null;
let isRunning = false;

app.get('/health', (req, res) => {
  res.json({ status: 'Server running', timestamp: new Date() });
});

app.post('/gemini/command', (req, res) => {
  const { command, action } = req.body;
  console.log(`Gemini Command: ${command} - ${action}`);
  res.json({ success: true, message: `Command sent: ${command}` });
});

app.post('/github/push', (req, res) => {
  const { repo, message } = req.body;
  console.log(`GitHub Push: ${repo} - ${message}`);
  res.json({ success: true, message: `GitHub push initiated` });
});

export const startServer = () => {
  if (isRunning) return { success: false, message: 'Already running' };
  
  server = app.listen(PORT, () => {
    isRunning = true;
    console.log(`Server running on localhost:${PORT}`);
  });
  
  return { success: true, message: 'Server started' };
};

export const stopServer = () => {
  if (!isRunning) return { success: false, message: 'Not running' };
  
  server.close(() => {
    isRunning = false;
    console.log('Server stopped');
  });
  
  return { success: true, message: 'Server stopped' };
};

export const getServerStatus = () => {
  return { isRunning, port: PORT };
};