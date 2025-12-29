import { describe, it, expect } from 'vitest';
import { getApiBaseUrl } from './api.js';

describe('API Utils', () => {
    describe('getApiBaseUrl', () => {
        it('should return the correct API base URL', () => {
            const baseUrl = getApiBaseUrl();
            expect(baseUrl).toBe('/api/v1');
        });

        it('should return a string', () => {
            const baseUrl = getApiBaseUrl();
            expect(typeof baseUrl).toBe('string');
        });

        it('should start with a forward slash', () => {
            const baseUrl = getApiBaseUrl();
            expect(baseUrl.startsWith('/')).toBe(true);
        });
    });
});
