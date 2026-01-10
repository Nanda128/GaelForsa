import React from 'react';

function LoadingOverlay({ show }) {
    if (!show) return null;

    return (
        <div className="loading-overlay" id="loading-overlay">
            <div className="loading-spinner"></div>
        </div>
    );
}

export default LoadingOverlay;

