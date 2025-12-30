import {describe, it, expect} from 'vitest';
import {escapeHtml, formatDateTime} from './helpers.js';

describe('Helper Utils', () => {
    describe('escapeHtml', () => {
        it('should escape HTML special characters', () => {
            const input = '<script>alert("xss")</script>';
            const result = escapeHtml(input);
            expect(result).not.toContain('<script>');
            expect(result).toContain('&lt;');
            expect(result).toContain('&gt;');
        });

        it('should return empty string for null input', () => {
            expect(escapeHtml(null)).toBe('');
        });

        it('should return empty string for undefined input', () => {
            expect(escapeHtml(undefined)).toBe('');
        });

        it('should return empty string for empty string input', () => {
            expect(escapeHtml('')).toBe('');
        });

        it('should not modify safe text', () => {
            const safeText = 'Hello World';
            expect(escapeHtml(safeText)).toBe('Hello World');
        });

        it('should escape ampersands', () => {
            const input = 'Tom & Jerry';
            const result = escapeHtml(input);
            expect(result).toContain('&amp;');
        });
    });

    describe('formatDateTime', () => {
        it('should return N/A for null input', () => {
            expect(formatDateTime(null)).toBe('N/A');
        });

        it('should return N/A for undefined input', () => {
            expect(formatDateTime(undefined)).toBe('N/A');
        });

        it('should return N/A for empty string', () => {
            expect(formatDateTime('')).toBe('N/A');
        });

        it('should format a valid ISO date string', () => {
            const isoString = '2025-06-15T14:30:00Z';
            const result = formatDateTime(isoString);

            expect(result).toContain('2025');
            expect(result).not.toBe('N/A');
        });

        it('should return a non-empty string for valid date', () => {
            const isoString = '2025-01-01T00:00:00Z';
            const result = formatDateTime(isoString);
            expect(result.length).toBeGreaterThan(0);
            expect(result).not.toBe('N/A');
        });
    });
});

