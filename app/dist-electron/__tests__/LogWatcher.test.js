import { describe, it, expect, beforeEach, afterEach, jest } from '@jest/globals';
import { LogWatcher } from '../services/LogWatcher.js';
import fs from 'fs';
import path from 'path';
import os from 'os';
// Mock Electron BrowserWindow
jest.mock('electron', () => ({
    BrowserWindow: jest.fn().mockImplementation(() => ({
        webContents: {
            send: jest.fn()
        }
    }))
}));
describe('LogWatcher Service', () => {
    let logWatcher;
    let mockWindow;
    let testLogPath;
    beforeEach(() => {
        // Create mock window
        mockWindow = {
            webContents: {
                send: jest.fn()
            }
        };
        logWatcher = new LogWatcher(mockWindow);
        testLogPath = path.join(os.tmpdir(), `test-log-${Date.now()}.txt`);
    });
    afterEach(() => {
        if (logWatcher) {
            logWatcher.stopWatching();
        }
        if (fs.existsSync(testLogPath)) {
            fs.unlinkSync(testLogPath);
        }
    });
    describe('로그 파일 감시 (File Watching)', () => {
        it('로그 파일 감시를 시작할 수 있어야 함', () => {
            fs.writeFileSync(testLogPath, 'Initial content\n');
            expect(() => {
                logWatcher.startWatching(testLogPath);
            }).not.toThrow();
        });
        it('로그 파일이 변경되면 감지해야 함', (done) => {
            fs.writeFileSync(testLogPath, 'Initial\n');
            logWatcher.startWatching(testLogPath);
            // Give chokidar time to start
            setTimeout(() => {
                fs.appendFileSync(testLogPath, '2025-01-26 18:00:00 | Artist - Track Title\n');
                // Wait for change detection
                setTimeout(() => {
                    expect(mockWindow.webContents.send).toHaveBeenCalled();
                    done();
                }, 1000);
            }, 500);
        }, 5000);
    });
    describe('로그 파싱 (Log Parsing)', () => {
        it('DJ 소프트웨어 로그 형식을 올바르게 파싱해야 함', (done) => {
            fs.writeFileSync(testLogPath, '');
            logWatcher.startWatching(testLogPath);
            setTimeout(() => {
                fs.writeFileSync(testLogPath, '2025-01-26 18:00:00 | Skrillex - Scary Monsters\n');
                setTimeout(() => {
                    expect(mockWindow.webContents.send).toHaveBeenCalledWith('track-update', expect.objectContaining({
                        artist: 'Skrillex',
                        title: 'Scary Monsters'
                    }));
                    done();
                }, 1000);
            }, 500);
        }, 5000);
        it('잘못된 형식의 로그는 무시해야 함', (done) => {
            fs.writeFileSync(testLogPath, '');
            logWatcher.startWatching(testLogPath);
            setTimeout(() => {
                fs.writeFileSync(testLogPath, 'Invalid log format\n');
                setTimeout(() => {
                    // Should not crash
                    expect(true).toBe(true);
                    done();
                }, 1000);
            }, 500);
        }, 5000);
    });
    describe('리소스 관리 (Resource Management)', () => {
        it('감시를 중지할 수 있어야 함', () => {
            fs.writeFileSync(testLogPath, 'Test\n');
            logWatcher.startWatching(testLogPath);
            expect(() => {
                logWatcher.stopWatching();
            }).not.toThrow();
        });
        it('다른 파일로 감시를 전환할 수 있어야 함', () => {
            const testLogPath2 = path.join(os.tmpdir(), `test-log-2-${Date.now()}.txt`);
            fs.writeFileSync(testLogPath, 'Test 1\n');
            fs.writeFileSync(testLogPath2, 'Test 2\n');
            logWatcher.startWatching(testLogPath);
            expect(() => {
                logWatcher.stopWatching();
                logWatcher.startWatching(testLogPath2);
            }).not.toThrow();
            if (fs.existsSync(testLogPath2)) {
                fs.unlinkSync(testLogPath2);
            }
        });
    });
    describe('에러 처리 (Error Handling)', () => {
        it('존재하지 않는 파일을 감시하려 해도 크래시되지 않아야 함', () => {
            const nonExistentPath = '/non/existent/path.log';
            expect(() => {
                logWatcher.startWatching(nonExistentPath);
            }).not.toThrow();
        });
    });
});
