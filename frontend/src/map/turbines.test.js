import {describe, it, expect} from 'vitest';
import {STATUS_COLORS, CLUSTER_CONFIG} from './turbines.js';

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

    describe('CLUSTER_CONFIG', () => {
        it('should enable chunked loading for performance', () => {
            expect(CLUSTER_CONFIG.chunkedLoading).toBe(true);
        });

        it('should have a reasonable max cluster radius', () => {
            expect(CLUSTER_CONFIG.maxClusterRadius).toBe(50);
            expect(CLUSTER_CONFIG.maxClusterRadius).toBeGreaterThan(0);
            expect(CLUSTER_CONFIG.maxClusterRadius).toBeLessThanOrEqual(100);
        });

        it('should enable spiderfy on max zoom', () => {
            expect(CLUSTER_CONFIG.spiderfyOnMaxZoom).toBe(true);
        });

        it('should disable coverage on hover for performance', () => {
            expect(CLUSTER_CONFIG.showCoverageOnHover).toBe(false);
        });

        it('should enable zoom to bounds on click', () => {
            expect(CLUSTER_CONFIG.zoomToBoundsOnClick).toBe(true);
        });

        it('should disable clustering at high zoom levels', () => {
            expect(CLUSTER_CONFIG.disableClusteringAtZoom).toBe(16);
            expect(CLUSTER_CONFIG.disableClusteringAtZoom).toBeGreaterThan(10);
        });

        it('should have all required clustering options', () => {
            const requiredKeys = [
                'chunkedLoading',
                'maxClusterRadius',
                'spiderfyOnMaxZoom',
                'showCoverageOnHover',
                'zoomToBoundsOnClick',
                'disableClusteringAtZoom'
            ];
            requiredKeys.forEach(key => {
                expect(CLUSTER_CONFIG).toHaveProperty(key);
            });
        });
    });
});

