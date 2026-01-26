import { spawn, ChildProcess } from 'child_process';
import path from 'path';

export class AIService {
  private pythonProcess: ChildProcess | null = null;

  constructor() {
    // Path to the python script
    // In production, this might need to be adjusted to point to a bundled executable or script
    // For dev, we assume it's in the ai/src folder relative to the project root
    // But since we are in app/electron/services, we need to go up quite a bit
    // app/electron/services -> app/electron -> app -> root -> ai/src
    // However, for simplicity in this plan, let's assume we pass the path or configure it.
  }

  start(scriptPath: string) {
    this.pythonProcess = spawn('python3', [scriptPath]);

    if (this.pythonProcess.stdout) {
      this.pythonProcess.stdout.on('data', (data) => {
        console.log(`AI Output: ${data}`);
        // Parse JSON output from Python and send to renderer if needed
      });
    }

    if (this.pythonProcess.stderr) {
      this.pythonProcess.stderr.on('data', (data) => {
        console.error(`AI Error: ${data}`);
      });
    }

    this.pythonProcess.on('close', (code) => {
      console.log(`AI process exited with code ${code}`);
    });
  }

  getRecommendation(currentTrack: any) {
    if (this.pythonProcess && this.pythonProcess.stdin) {
      const payload = JSON.stringify({ type: 'recommend', track: currentTrack }) + '\n';
      this.pythonProcess.stdin.write(payload);
    }
  }

  stop() {
    if (this.pythonProcess) {
      this.pythonProcess.kill();
      this.pythonProcess = null;
    }
  }
}
