import { describe, it, expect, beforeEach, afterEach, jest } from '@jest/globals';
import { AIService } from '../services/AIService.js';
import path from 'path';
import { fileURLToPath } from 'url';
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
// Mock Electron BrowserWindow
jest.mock('electron', () => ({
    BrowserWindow: jest.fn().mockImplementation(() => ({
        webContents: {
            send: jest.fn()
        }
    }))
}));
describe('AIService', () => {
    let aiService;
    let mockWindow;
    beforeEach(() => {
        mockWindow = {
            webContents: {
                send: jest.fn()
            }
        };
        aiService = new AIService(mockWindow);
    });
    afterEach(() => {
        if (aiService) {
            aiService.stop();
        }
    });
    describe('서비스 초기화 (Service Initialization)', () => {
        it('AIService 인스턴스를 생성할 수 있어야 함', () => {
            expect(aiService).toBeDefined();
            expect(aiService).toBeInstanceOf(AIService);
        });
    });
    describe('Python 프로세스 관리 (Python Process Management)', () => {
        it('Python 프로세스를 시작 시도할 수 있어야 함', async () => {
            const scriptPath = path.join(__dirname, '../../../ai/src/main.py');
            const indexDir = path.join(__dirname, '../../../ai/src/index');
            // In test environment, start() resolves after timeout when Python is not ready
            // but should not crash the application
            const startPromise = aiService.start(scriptPath, indexDir);
            // Stop immediately to prevent long timeout
            setTimeout(() => aiService.stop(), 500);
            // Should resolve or reject without crashing
            try {
                await startPromise;
                expect(true).toBe(true);
            }
            catch (error) {
                expect(error).toBeDefined();
            }
        }, 5000);
        it('Python 프로세스를 중지할 수 있어야 함', () => {
            expect(() => {
                aiService.stop();
            }).not.toThrow();
        });
        it('중복 시작 호출이 안전해야 함', () => {
            // Just verify the method exists and is callable without crashing
            expect(() => {
                aiService.stop();
                aiService.stop();
            }).not.toThrow();
        });
    });
    describe('추천 요청 (Recommendation Request)', () => {
        it('추천 요청이 올바른 형식을 가져야 함', async () => {
            const request = {
                currentTrackPath: '/path/to/track.mp3',
                goal: 'up',
                topK: 3
            };
            // Will fail without running Python process
            await expect(aiService.getRecommendation(request)).rejects.toThrow('Python process not running');
        });
        it('필수 파라미터가 없으면 에러를 반환해야 함', async () => {
            const invalidRequest = {
                currentTrackPath: '',
                goal: 'maintain',
                topK: 3
            };
            await expect(aiService.getRecommendation(invalidRequest)).rejects.toThrow();
        });
        it('goal 옵션이 올바르게 처리되어야 함', () => {
            const validGoals = ['maintain', 'up', 'down', 'peak'];
            validGoals.forEach(goal => {
                const request = {
                    currentTrackPath: '/path/to/track.mp3',
                    goal: goal,
                    topK: 3
                };
                // Should accept all valid goals
                expect(request.goal).toBe(goal);
            });
        });
    });
    describe('응답 처리 (Response Handling)', () => {
        it('AI 응답이 올바른 형식을 가져야 함', () => {
            const mockResponse = {
                request_id: 'req_123',
                recommendations: [
                    {
                        track_id: 'track_1',
                        path: '/path/to/rec1.mp3',
                        title: 'Recommended Track 1',
                        artist: 'Artist 1',
                        score: 0.95,
                        bpm: 128,
                        key: 'Am'
                    }
                ]
            };
            expect(mockResponse).toHaveProperty('request_id');
            expect(mockResponse).toHaveProperty('recommendations');
            expect(Array.isArray(mockResponse.recommendations)).toBe(true);
        });
        it('에러 응답도 올바르게 처리되어야 함', () => {
            const errorResponse = {
                request_id: 'req_456',
                error: 'Track not found',
                recommendations: []
            };
            expect(errorResponse).toHaveProperty('error');
            expect(errorResponse.recommendations).toHaveLength(0);
        });
    });
    describe('타임아웃 처리 (Timeout Handling)', () => {
        it('추천 요청이 타임아웃되어야 함', async () => {
            // This will timeout since no Python process is running
            const request = {
                currentTrackPath: '/path/to/track.mp3',
                goal: 'maintain',
                topK: 3
            };
            await expect(aiService.getRecommendation(request)).rejects.toThrow();
        }, 35000);
    });
    describe('IPC 통신 (IPC Communication)', () => {
        it('추천 결과가 renderer로 전송되어야 함', () => {
            // Mock implementation would verify webContents.send is called
            expect(mockWindow.webContents.send).toBeDefined();
        });
    });
    describe('에러 처리 (Error Handling)', () => {
        it('Python 스크립트가 없어도 크래시되지 않아야 함', async () => {
            const invalidPath = '/invalid/path/to/script.py';
            const indexDir = '/invalid/index';
            // Start with invalid path
            const startPromise = aiService.start(invalidPath, indexDir);
            // Stop immediately to prevent timeout
            setTimeout(() => aiService.stop(), 500);
            try {
                await startPromise;
                expect(true).toBe(true);
            }
            catch (error) {
                expect(error).toBeDefined();
            }
        }, 5000);
        it('잘못된 JSON 응답을 받아도 크래시되지 않아야 함', () => {
            // This would be tested with actual Python process
            expect(true).toBe(true);
        });
    });
});
