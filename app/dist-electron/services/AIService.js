import { spawn } from 'child_process';
export class AIService {
    pythonProcess = null;
    mainWindow = null;
    pendingRequests = new Map();
    buffer = '';
    requestIdCounter = 0;
    constructor(mainWindow) {
        this.mainWindow = mainWindow;
    }
    start(scriptPath, indexDir) {
        return new Promise((resolve, reject) => {
            if (this.pythonProcess) {
                console.log('Python process already running');
                resolve();
                return;
            }
            console.log(`Starting AI service with script: ${scriptPath}, index: ${indexDir}`);
            let resolved = false;
            const safeResolve = () => {
                if (!resolved) {
                    resolved = true;
                    resolve();
                }
            };
            const safeReject = (error) => {
                if (!resolved) {
                    resolved = true;
                    reject(error);
                }
            };
            this.pythonProcess = spawn('python3', [
                scriptPath,
                'serve',
                '--index_dir', indexDir,
                '--top_k', '3'
            ], {
                stdio: ['pipe', 'pipe', 'pipe']
            });
            if (this.pythonProcess.stdout) {
                this.pythonProcess.stdout.on('data', (data) => {
                    this.handleStdout(data);
                });
            }
            if (this.pythonProcess.stderr) {
                this.pythonProcess.stderr.on('data', (data) => {
                    const errorMsg = data.toString();
                    console.error(`AI Error: ${errorMsg}`);
                    // Check for startup messages
                    if (errorMsg.includes('Ready') || errorMsg.includes('Loaded index')) {
                        console.log('AI service ready');
                        safeResolve();
                    }
                });
            }
            this.pythonProcess.on('error', (error) => {
                console.error('Failed to start Python process:', error);
                this.pythonProcess = null;
                safeReject(error);
            });
            this.pythonProcess.on('close', (code) => {
                console.log(`AI process exited with code ${code}`);
                this.pythonProcess = null;
                // Reject all pending requests
                for (const [id, { reject: reqReject, timeout }] of this.pendingRequests.entries()) {
                    clearTimeout(timeout);
                    reqReject(new Error('Python process terminated'));
                }
                this.pendingRequests.clear();
                // Reject start promise if process exited with error
                if (code !== 0) {
                    safeReject(new Error(`Python process exited with code ${code}`));
                }
                else {
                    safeResolve();
                }
            });
            // Timeout for startup
            setTimeout(() => {
                if (this.pythonProcess) {
                    console.log('AI service started (timeout fallback)');
                    safeResolve();
                }
            }, 5000);
        });
    }
    handleStdout(data) {
        this.buffer += data.toString();
        const lines = this.buffer.split('\n');
        this.buffer = lines.pop() || '';
        for (const line of lines) {
            if (!line.trim())
                continue;
            try {
                const response = JSON.parse(line);
                console.log('AI Response:', response);
                // Handle response
                if (response.request_id && this.pendingRequests.has(response.request_id)) {
                    const { resolve, timeout } = this.pendingRequests.get(response.request_id);
                    clearTimeout(timeout);
                    this.pendingRequests.delete(response.request_id);
                    if (response.error) {
                        resolve({ error: response.error, recommendations: [] });
                    }
                    else {
                        resolve(response);
                    }
                    // Send to renderer
                    if (this.mainWindow) {
                        this.mainWindow.webContents.send('recommendations-update', response);
                    }
                }
            }
            catch (error) {
                console.error('Failed to parse AI response:', line, error);
            }
        }
    }
    async getRecommendation(request) {
        if (!this.pythonProcess || !this.pythonProcess.stdin) {
            throw new Error('Python process not running');
        }
        const requestId = `req_${this.requestIdCounter++}`;
        return new Promise((resolve, reject) => {
            const timeout = setTimeout(() => {
                this.pendingRequests.delete(requestId);
                reject(new Error('AI recommendation timeout'));
            }, 30000); // 30 second timeout
            this.pendingRequests.set(requestId, { resolve, reject, timeout });
            const payload = {
                request_id: requestId,
                type: 'recommend',
                current: request.currentTrackPath,
                goal: request.goal || 'maintain',
                top_k: request.topK || 3
            };
            try {
                this.pythonProcess.stdin.write(JSON.stringify(payload) + '\n');
            }
            catch (error) {
                clearTimeout(timeout);
                this.pendingRequests.delete(requestId);
                reject(error);
            }
        });
    }
    stop() {
        if (this.pythonProcess) {
            console.log('Stopping AI service...');
            this.pythonProcess.kill('SIGTERM');
            // Force kill after 5 seconds
            setTimeout(() => {
                if (this.pythonProcess) {
                    this.pythonProcess.kill('SIGKILL');
                }
            }, 5000);
            this.pythonProcess = null;
        }
        // Clear all pending requests
        for (const [id, { reject, timeout }] of this.pendingRequests.entries()) {
            clearTimeout(timeout);
            reject(new Error('AI service stopped'));
        }
        this.pendingRequests.clear();
    }
}
