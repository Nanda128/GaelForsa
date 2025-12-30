// General Helper functions

/**
 * Escape HTML to prevent XSS
 * @param {string} text - The text to escape
 * @returns {string} The escaped text
 */
export function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/**
 * Format ISO datetime to readable format
 * @param {string} isoString - ISO date string
 * @returns {string} Formatted date string
 */
export function formatDateTime(isoString) {
    if (!isoString) return 'N/A';

    const date = new Date(isoString);
    return date.toLocaleString('en-IE', {
        year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'
    });
}
