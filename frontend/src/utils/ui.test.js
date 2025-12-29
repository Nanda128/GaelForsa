import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { showLoading, hideLoading, showError } from './ui.js';

describe('UI Utils', () => {
    beforeEach(() => {
        document.body.innerHTML = '';
    });

    afterEach(() => {
        document.body.innerHTML = '';
        vi.clearAllTimers();
    });

    describe('showLoading', () => {
        it('should add loading overlay to the document body', () => {
            showLoading();
            const overlay = document.getElementById('loading-overlay');
            expect(overlay).not.toBeNull();
        });

        it('should have the correct class name', () => {
            showLoading();
            const overlay = document.getElementById('loading-overlay');
            expect(overlay.className).toBe('loading-overlay');
        });

        it('should contain a loading spinner', () => {
            showLoading();
            const spinner = document.querySelector('.loading-spinner');
            expect(spinner).not.toBeNull();
        });

        it('should not create multiple overlays when called twice', () => {
            showLoading();
            showLoading();
            const overlays = document.querySelectorAll('#loading-overlay');
            expect(overlays.length).toBe(1);
        });
    });

    describe('hideLoading', () => {
        it('should remove the loading overlay', () => {
            showLoading();
            hideLoading();
            const overlay = document.getElementById('loading-overlay');
            expect(overlay).toBeNull();
        });

        it('should not throw error if overlay does not exist', () => {
            expect(() => hideLoading()).not.toThrow();
        });
    });

    describe('showError', () => {
        beforeEach(() => {
            vi.useFakeTimers();
        });

        afterEach(() => {
            vi.useRealTimers();
        });

        it('should add error message to the document body', () => {
            showError('Test error message');
            const errorDiv = document.querySelector('.error-message');
            expect(errorDiv).not.toBeNull();
        });

        it('should display the correct error message', () => {
            const message = 'Something went wrong!';
            showError(message);
            const errorDiv = document.querySelector('.error-message');
            expect(errorDiv.textContent).toBe(message);
        });

        it('should remove error after default duration', () => {
            showError('Test error');

            vi.advanceTimersByTime(5000);

            const errorDiv = document.querySelector('.error-message');
            expect(errorDiv).toBeNull();
        });

        it('should remove error after custom duration', () => {
            showError('Test error', 2000);

            vi.advanceTimersByTime(2000);

            const errorDiv = document.querySelector('.error-message');
            expect(errorDiv).toBeNull();
        });

        it('should still show error before duration elapses', () => {
            showError('Test error', 5000);

            vi.advanceTimersByTime(3000);

            const errorDiv = document.querySelector('.error-message');
            expect(errorDiv).not.toBeNull();
        });
    });
});

