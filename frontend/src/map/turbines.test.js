import {describe, it, expect} from 'vitest';
import {STATUS_COLORS} from './turbines.js';

describe('Turbines Module', () => {
    describe('STATUS_COLORS', () => {
        it('should have green status color', () => {
            expect(STATUS_COLORS.green).toBe('#28a745');
        });

        it('should have yellow status color', () => {
            expect(STATUS_COLORS.yellow).toBe('#ffc107');
        });

        it('should have red status color', () => {
            expect(STATUS_COLORS.red).toBe('#dc3545');
        });

        it('should have exactly 3 status colors', () => {
            expect(Object.keys(STATUS_COLORS).length).toBe(3);
        });

        it('should only contain valid hex color codes', () => {
            const hexColorPattern = /^#[0-9A-Fa-f]{6}$/;
            Object.values(STATUS_COLORS).forEach(color => {
                expect(color).toMatch(hexColorPattern);
            });
        });
    });
});

