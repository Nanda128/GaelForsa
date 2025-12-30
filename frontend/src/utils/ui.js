// UI Utilities

/**
 * Show loading overlay
 */
export function showLoading() {
    if (document.getElementById('loading-overlay')) return;

    const overlay = document.createElement('div');
    overlay.className = 'loading-overlay';
    overlay.id = 'loading-overlay';
    overlay.innerHTML = '<div class="loading-spinner"></div>';
    document.body.appendChild(overlay);
}

/**
 * Hide loading overlay
 */
export function hideLoading() {
    const overlay = document.getElementById('loading-overlay');
    if (overlay) {
        overlay.remove();
    }
}

/**
 * Show error message
 * @param {string} message - The error message to display
 * @param {number} duration - How long to show the message (ms)
 */
export function showError(message, duration = 5000) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.textContent = message;
    document.body.appendChild(errorDiv);

    // Auto-hide after duration
    setTimeout(() => {
        errorDiv.remove();
    }, duration);
}
